import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1630
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1628
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1629

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult232225
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def rawTerms : List Term := Proof.Events907.exact232225RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232225
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232225.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232222) (rightBinding := 232223)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7316⟩) (rightExpression := ⟨7216⟩)
    (transferEvent := 232224)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232221.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232162.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232225

namespace SemanticResult232229
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def rawTerms : List Term := Proof.Events907.exact232229RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232229
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232229.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232226) (rightBinding := 232227)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7317⟩) (rightExpression := ⟨7218⟩)
    (transferEvent := 232228)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232225.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232159.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232229

namespace SemanticResult232233
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def rawTerms : List Term := Proof.Events907.exact232233RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232233
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232233.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232230) (rightBinding := 232231)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7318⟩) (rightExpression := ⟨7220⟩)
    (transferEvent := 232232)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232229.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232156.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232233

namespace SemanticResult232237
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def rawTerms : List Term := Proof.Events907.exact232237RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232237
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232237.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232234) (rightBinding := 232235)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7319⟩) (rightExpression := ⟨7222⟩)
    (transferEvent := 232236)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232233.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232153.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232237

namespace SemanticResult232241
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def rawTerms : List Term := Proof.Events907.exact232241RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232241
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232241.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232238) (rightBinding := 232239)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7320⟩) (rightExpression := ⟨7224⟩)
    (transferEvent := 232240)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232237.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232150.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232241

namespace SemanticResult232245
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def rawTerms : List Term := Proof.Events907.exact232245RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232245
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232245.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232242) (rightBinding := 232243)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7321⟩) (rightExpression := ⟨7226⟩)
    (transferEvent := 232244)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232241.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232147.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232245

namespace SemanticResult232249
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def rawTerms : List Term := Proof.Events907.exact232249RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232249
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232249.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232246) (rightBinding := 232247)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7322⟩) (rightExpression := ⟨7228⟩)
    (transferEvent := 232248)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232245.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232144.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232249

namespace SemanticResult232253
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def rawTerms : List Term := Proof.Events907.exact232253RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232253
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232253.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232250) (rightBinding := 232251)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7323⟩) (rightExpression := ⟨7230⟩)
    (transferEvent := 232252)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232249.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232141.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232253

namespace SemanticResult232257
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def rawTerms : List Term := Proof.Events907.exact232257RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232257
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232257.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232254) (rightBinding := 232255)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7324⟩) (rightExpression := ⟨7232⟩)
    (transferEvent := 232256)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232253.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232138.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232257

namespace SemanticResult232261
def owner : Owner := ⟨.program ⟨257⟩, ⟨69086⟩⟩
def rawTerms : List Term := Proof.Events907.exact232261RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232261
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232261.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232258) (rightBinding := 232259)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7325⟩) (rightExpression := ⟨69085⟩)
    (transferEvent := 232260)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232257.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232135.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232261

namespace SemanticResult232337
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def rawTerms : List Term := Proof.Events907.exact232337RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232337
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232337.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge232265.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge232265.frameStart)
    (transferEvent := 232264) (owner := owner)
    (leftResult := 232261) (rightResult := 232102)
    (working := LeftOperatorMerge232265.working)
    (reconstruction := LeftOperatorMerge232265.reconstruction)
    (leftReference := .predecessor 0 232262 .coefficient) (rightReference := .predecessor 1 232263 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult232261.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232102.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge232265.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232284 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge232265.working
    [{ coefficient := (-1), key := LeftRelationMerge232284.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge232284.frameStart
      LeftRelationMerge232284.owner (.relation 232284) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge232284.deltas
    rows := LeftRelationMerge232284.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge232265.working LeftRelationMerge232284.source
        (relationContext LeftRelationMerge232284.source
          LeftRelationMerge232284.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge232265.working, LeftRelationMerge232284.deltas,
    LeftRelationMerge232284.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232284)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨48350⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge232265.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact mergeClaim selector selectorLower selectorUpper witness
  · exact relationAgreement0
  · decide +kernel
theorem relationApplicationAt1 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232287 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking1 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }]
def relationRhsRaw1 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase1 : Polynomial Owner :=
  subtract relationWorking0
    [{ coefficient := (-1), key := LeftRelationMerge232287.source }]
def relationReconstruction1 :
    MergeReconstructionAt history LeftRelationMerge232287.frameStart
      LeftRelationMerge232287.owner (.relation 232287) relationBase1
      relationWorking1 :=
  { deltas := LeftRelationMerge232287.deltas
    rows := LeftRelationMerge232287.rows
    agreement := by decide +kernel }
theorem relationAgreement1 :
    CanonicalAgreement (add relationBase1 relationReconstruction1.deltas)
      (relationPoly relationWorking0 LeftRelationMerge232287.source
        (relationContext LeftRelationMerge232287.source
          LeftRelationMerge232287.source.centralFactors 0 2) (-1)
        (relationRhsRaw1.map Term.toExact)) := by
  dsimp [relationReconstruction1, relationBase1, relationWorking1,
    relationRhsRaw1, relationWorking0, LeftRelationMerge232287.deltas,
    LeftRelationMerge232287.source]
  decide +kernel
theorem relationClaim1 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking1 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232287)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨45670⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw1)
    (accumulator := relationWorking0) (working := relationWorking1)
    (reconstruction := relationReconstruction1)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt1 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim0 selector selectorLower selectorUpper witness
  · exact relationAgreement1
  · decide +kernel
theorem relationApplicationAt2 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232290 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking2 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }]
def relationRhsRaw2 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase2 : Polynomial Owner :=
  subtract relationWorking1
    [{ coefficient := (-1), key := LeftRelationMerge232290.source }]
def relationReconstruction2 :
    MergeReconstructionAt history LeftRelationMerge232290.frameStart
      LeftRelationMerge232290.owner (.relation 232290) relationBase2
      relationWorking2 :=
  { deltas := LeftRelationMerge232290.deltas
    rows := LeftRelationMerge232290.rows
    agreement := by decide +kernel }
theorem relationAgreement2 :
    CanonicalAgreement (add relationBase2 relationReconstruction2.deltas)
      (relationPoly relationWorking1 LeftRelationMerge232290.source
        (relationContext LeftRelationMerge232290.source
          LeftRelationMerge232290.source.centralFactors 0 2) (-1)
        (relationRhsRaw2.map Term.toExact)) := by
  dsimp [relationReconstruction2, relationBase2, relationWorking2,
    relationRhsRaw2, relationWorking1, LeftRelationMerge232290.deltas,
    LeftRelationMerge232290.source]
  decide +kernel
theorem relationClaim2 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking2 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232290)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨42986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw2)
    (accumulator := relationWorking1) (working := relationWorking2)
    (reconstruction := relationReconstruction2)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt2 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim1 selector selectorLower selectorUpper witness
  · exact relationAgreement2
  · decide +kernel
theorem relationApplicationAt3 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232293 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking3 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }]
def relationRhsRaw3 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase3 : Polynomial Owner :=
  subtract relationWorking2
    [{ coefficient := (-1), key := LeftRelationMerge232293.source }]
def relationReconstruction3 :
    MergeReconstructionAt history LeftRelationMerge232293.frameStart
      LeftRelationMerge232293.owner (.relation 232293) relationBase3
      relationWorking3 :=
  { deltas := LeftRelationMerge232293.deltas
    rows := LeftRelationMerge232293.rows
    agreement := by decide +kernel }
theorem relationAgreement3 :
    CanonicalAgreement (add relationBase3 relationReconstruction3.deltas)
      (relationPoly relationWorking2 LeftRelationMerge232293.source
        (relationContext LeftRelationMerge232293.source
          LeftRelationMerge232293.source.centralFactors 0 2) (-1)
        (relationRhsRaw3.map Term.toExact)) := by
  dsimp [relationReconstruction3, relationBase3, relationWorking3,
    relationRhsRaw3, relationWorking2, LeftRelationMerge232293.deltas,
    LeftRelationMerge232293.source]
  decide +kernel
theorem relationClaim3 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking3 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232293)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨40306⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw3)
    (accumulator := relationWorking2) (working := relationWorking3)
    (reconstruction := relationReconstruction3)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt3 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim2 selector selectorLower selectorUpper witness
  · exact relationAgreement3
  · decide +kernel
theorem relationApplicationAt4 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232296 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking4 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }]
def relationRhsRaw4 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase4 : Polynomial Owner :=
  subtract relationWorking3
    [{ coefficient := (-1), key := LeftRelationMerge232296.source }]
def relationReconstruction4 :
    MergeReconstructionAt history LeftRelationMerge232296.frameStart
      LeftRelationMerge232296.owner (.relation 232296) relationBase4
      relationWorking4 :=
  { deltas := LeftRelationMerge232296.deltas
    rows := LeftRelationMerge232296.rows
    agreement := by decide +kernel }
theorem relationAgreement4 :
    CanonicalAgreement (add relationBase4 relationReconstruction4.deltas)
      (relationPoly relationWorking3 LeftRelationMerge232296.source
        (relationContext LeftRelationMerge232296.source
          LeftRelationMerge232296.source.centralFactors 0 2) (-1)
        (relationRhsRaw4.map Term.toExact)) := by
  dsimp [relationReconstruction4, relationBase4, relationWorking4,
    relationRhsRaw4, relationWorking3, LeftRelationMerge232296.deltas,
    LeftRelationMerge232296.source]
  decide +kernel
theorem relationClaim4 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking4 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232296)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨37630⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw4)
    (accumulator := relationWorking3) (working := relationWorking4)
    (reconstruction := relationReconstruction4)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt4 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim3 selector selectorLower selectorUpper witness
  · exact relationAgreement4
  · decide +kernel
theorem relationApplicationAt5 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232299 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking5 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }]
def relationRhsRaw5 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase5 : Polynomial Owner :=
  subtract relationWorking4
    [{ coefficient := (-1), key := LeftRelationMerge232299.source }]
def relationReconstruction5 :
    MergeReconstructionAt history LeftRelationMerge232299.frameStart
      LeftRelationMerge232299.owner (.relation 232299) relationBase5
      relationWorking5 :=
  { deltas := LeftRelationMerge232299.deltas
    rows := LeftRelationMerge232299.rows
    agreement := by decide +kernel }
theorem relationAgreement5 :
    CanonicalAgreement (add relationBase5 relationReconstruction5.deltas)
      (relationPoly relationWorking4 LeftRelationMerge232299.source
        (relationContext LeftRelationMerge232299.source
          LeftRelationMerge232299.source.centralFactors 0 2) (-1)
        (relationRhsRaw5.map Term.toExact)) := by
  dsimp [relationReconstruction5, relationBase5, relationWorking5,
    relationRhsRaw5, relationWorking4, LeftRelationMerge232299.deltas,
    LeftRelationMerge232299.source]
  decide +kernel
theorem relationClaim5 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking5 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232299)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨34950⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw5)
    (accumulator := relationWorking4) (working := relationWorking5)
    (reconstruction := relationReconstruction5)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt5 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim4 selector selectorLower selectorUpper witness
  · exact relationAgreement5
  · decide +kernel
theorem relationApplicationAt6 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232302 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking6 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }]
def relationRhsRaw6 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase6 : Polynomial Owner :=
  subtract relationWorking5
    [{ coefficient := (-1), key := LeftRelationMerge232302.source }]
def relationReconstruction6 :
    MergeReconstructionAt history LeftRelationMerge232302.frameStart
      LeftRelationMerge232302.owner (.relation 232302) relationBase6
      relationWorking6 :=
  { deltas := LeftRelationMerge232302.deltas
    rows := LeftRelationMerge232302.rows
    agreement := by decide +kernel }
theorem relationAgreement6 :
    CanonicalAgreement (add relationBase6 relationReconstruction6.deltas)
      (relationPoly relationWorking5 LeftRelationMerge232302.source
        (relationContext LeftRelationMerge232302.source
          LeftRelationMerge232302.source.centralFactors 0 2) (-1)
        (relationRhsRaw6.map Term.toExact)) := by
  dsimp [relationReconstruction6, relationBase6, relationWorking6,
    relationRhsRaw6, relationWorking5, LeftRelationMerge232302.deltas,
    LeftRelationMerge232302.source]
  decide +kernel
theorem relationClaim6 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking6 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232302)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨29286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw6)
    (accumulator := relationWorking5) (working := relationWorking6)
    (reconstruction := relationReconstruction6)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt6 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim5 selector selectorLower selectorUpper witness
  · exact relationAgreement6
  · decide +kernel
theorem relationApplicationAt7 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232305 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking7 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }]
def relationRhsRaw7 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase7 : Polynomial Owner :=
  subtract relationWorking6
    [{ coefficient := (-1), key := LeftRelationMerge232305.source }]
def relationReconstruction7 :
    MergeReconstructionAt history LeftRelationMerge232305.frameStart
      LeftRelationMerge232305.owner (.relation 232305) relationBase7
      relationWorking7 :=
  { deltas := LeftRelationMerge232305.deltas
    rows := LeftRelationMerge232305.rows
    agreement := by decide +kernel }
theorem relationAgreement7 :
    CanonicalAgreement (add relationBase7 relationReconstruction7.deltas)
      (relationPoly relationWorking6 LeftRelationMerge232305.source
        (relationContext LeftRelationMerge232305.source
          LeftRelationMerge232305.source.centralFactors 0 2) (-1)
        (relationRhsRaw7.map Term.toExact)) := by
  dsimp [relationReconstruction7, relationBase7, relationWorking7,
    relationRhsRaw7, relationWorking6, LeftRelationMerge232305.deltas,
    LeftRelationMerge232305.source]
  decide +kernel
theorem relationClaim7 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking7 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232305)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨26606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw7)
    (accumulator := relationWorking6) (working := relationWorking7)
    (reconstruction := relationReconstruction7)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt7 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim6 selector selectorLower selectorUpper witness
  · exact relationAgreement7
  · decide +kernel
theorem relationApplicationAt8 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232308 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking8 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw8 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase8 : Polynomial Owner :=
  subtract relationWorking7
    [{ coefficient := (-1), key := LeftRelationMerge232308.source }]
def relationReconstruction8 :
    MergeReconstructionAt history LeftRelationMerge232308.frameStart
      LeftRelationMerge232308.owner (.relation 232308) relationBase8
      relationWorking8 :=
  { deltas := LeftRelationMerge232308.deltas
    rows := LeftRelationMerge232308.rows
    agreement := by decide +kernel }
theorem relationAgreement8 :
    CanonicalAgreement (add relationBase8 relationReconstruction8.deltas)
      (relationPoly relationWorking7 LeftRelationMerge232308.source
        (relationContext LeftRelationMerge232308.source
          LeftRelationMerge232308.source.centralFactors 0 2) (-1)
        (relationRhsRaw8.map Term.toExact)) := by
  dsimp [relationReconstruction8, relationBase8, relationWorking8,
    relationRhsRaw8, relationWorking7, LeftRelationMerge232308.deltas,
    LeftRelationMerge232308.source]
  decide +kernel
theorem relationClaim8 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking8 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232308)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨66531⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw8)
    (accumulator := relationWorking7) (working := relationWorking8)
    (reconstruction := relationReconstruction8)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt8 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim7 selector selectorLower selectorUpper witness
  · exact relationAgreement8
  · decide +kernel
theorem relationApplicationAt9 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232311 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking9 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw9 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase9 : Polynomial Owner :=
  subtract relationWorking8
    [{ coefficient := (-1), key := LeftRelationMerge232311.source }]
def relationReconstruction9 :
    MergeReconstructionAt history LeftRelationMerge232311.frameStart
      LeftRelationMerge232311.owner (.relation 232311) relationBase9
      relationWorking9 :=
  { deltas := LeftRelationMerge232311.deltas
    rows := LeftRelationMerge232311.rows
    agreement := by decide +kernel }
theorem relationAgreement9 :
    CanonicalAgreement (add relationBase9 relationReconstruction9.deltas)
      (relationPoly relationWorking8 LeftRelationMerge232311.source
        (relationContext LeftRelationMerge232311.source
          LeftRelationMerge232311.source.centralFactors 0 2) (-1)
        (relationRhsRaw9.map Term.toExact)) := by
  dsimp [relationReconstruction9, relationBase9, relationWorking9,
    relationRhsRaw9, relationWorking8, LeftRelationMerge232311.deltas,
    LeftRelationMerge232311.source]
  decide +kernel
theorem relationClaim9 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking9 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232311)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨63062⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw9)
    (accumulator := relationWorking8) (working := relationWorking9)
    (reconstruction := relationReconstruction9)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt9 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim8 selector selectorLower selectorUpper witness
  · exact relationAgreement9
  · decide +kernel
theorem relationApplicationAt10 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232314 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking10 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw10 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase10 : Polynomial Owner :=
  subtract relationWorking9
    [{ coefficient := (-1), key := LeftRelationMerge232314.source }]
def relationReconstruction10 :
    MergeReconstructionAt history LeftRelationMerge232314.frameStart
      LeftRelationMerge232314.owner (.relation 232314) relationBase10
      relationWorking10 :=
  { deltas := LeftRelationMerge232314.deltas
    rows := LeftRelationMerge232314.rows
    agreement := by decide +kernel }
theorem relationAgreement10 :
    CanonicalAgreement (add relationBase10 relationReconstruction10.deltas)
      (relationPoly relationWorking9 LeftRelationMerge232314.source
        (relationContext LeftRelationMerge232314.source
          LeftRelationMerge232314.source.centralFactors 0 2) (-1)
        (relationRhsRaw10.map Term.toExact)) := by
  dsimp [relationReconstruction10, relationBase10, relationWorking10,
    relationRhsRaw10, relationWorking9, LeftRelationMerge232314.deltas,
    LeftRelationMerge232314.source]
  decide +kernel
theorem relationClaim10 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking10 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232314)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨60082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw10)
    (accumulator := relationWorking9) (working := relationWorking10)
    (reconstruction := relationReconstruction10)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt10 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim9 selector selectorLower selectorUpper witness
  · exact relationAgreement10
  · decide +kernel
theorem relationApplicationAt11 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232317 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking11 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw11 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase11 : Polynomial Owner :=
  subtract relationWorking10
    [{ coefficient := (-1), key := LeftRelationMerge232317.source }]
def relationReconstruction11 :
    MergeReconstructionAt history LeftRelationMerge232317.frameStart
      LeftRelationMerge232317.owner (.relation 232317) relationBase11
      relationWorking11 :=
  { deltas := LeftRelationMerge232317.deltas
    rows := LeftRelationMerge232317.rows
    agreement := by decide +kernel }
theorem relationAgreement11 :
    CanonicalAgreement (add relationBase11 relationReconstruction11.deltas)
      (relationPoly relationWorking10 LeftRelationMerge232317.source
        (relationContext LeftRelationMerge232317.source
          LeftRelationMerge232317.source.centralFactors 0 2) (-1)
        (relationRhsRaw11.map Term.toExact)) := by
  dsimp [relationReconstruction11, relationBase11, relationWorking11,
    relationRhsRaw11, relationWorking10, LeftRelationMerge232317.deltas,
    LeftRelationMerge232317.source]
  decide +kernel
theorem relationClaim11 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking11 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232317)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨57102⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw11)
    (accumulator := relationWorking10) (working := relationWorking11)
    (reconstruction := relationReconstruction11)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt11 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim10 selector selectorLower selectorUpper witness
  · exact relationAgreement11
  · decide +kernel
theorem relationApplicationAt12 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232320 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking12 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw12 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase12 : Polynomial Owner :=
  subtract relationWorking11
    [{ coefficient := (-1), key := LeftRelationMerge232320.source }]
def relationReconstruction12 :
    MergeReconstructionAt history LeftRelationMerge232320.frameStart
      LeftRelationMerge232320.owner (.relation 232320) relationBase12
      relationWorking12 :=
  { deltas := LeftRelationMerge232320.deltas
    rows := LeftRelationMerge232320.rows
    agreement := by decide +kernel }
theorem relationAgreement12 :
    CanonicalAgreement (add relationBase12 relationReconstruction12.deltas)
      (relationPoly relationWorking11 LeftRelationMerge232320.source
        (relationContext LeftRelationMerge232320.source
          LeftRelationMerge232320.source.centralFactors 0 2) (-1)
        (relationRhsRaw12.map Term.toExact)) := by
  dsimp [relationReconstruction12, relationBase12, relationWorking12,
    relationRhsRaw12, relationWorking11, LeftRelationMerge232320.deltas,
    LeftRelationMerge232320.source]
  decide +kernel
theorem relationClaim12 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking12 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232320)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨54122⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw12)
    (accumulator := relationWorking11) (working := relationWorking12)
    (reconstruction := relationReconstruction12)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt12 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim11 selector selectorLower selectorUpper witness
  · exact relationAgreement12
  · decide +kernel
theorem relationApplicationAt13 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232323 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking13 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw13 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase13 : Polynomial Owner :=
  subtract relationWorking12
    [{ coefficient := (-1), key := LeftRelationMerge232323.source }]
def relationReconstruction13 :
    MergeReconstructionAt history LeftRelationMerge232323.frameStart
      LeftRelationMerge232323.owner (.relation 232323) relationBase13
      relationWorking13 :=
  { deltas := LeftRelationMerge232323.deltas
    rows := LeftRelationMerge232323.rows
    agreement := by decide +kernel }
theorem relationAgreement13 :
    CanonicalAgreement (add relationBase13 relationReconstruction13.deltas)
      (relationPoly relationWorking12 LeftRelationMerge232323.source
        (relationContext LeftRelationMerge232323.source
          LeftRelationMerge232323.source.centralFactors 0 2) (-1)
        (relationRhsRaw13.map Term.toExact)) := by
  dsimp [relationReconstruction13, relationBase13, relationWorking13,
    relationRhsRaw13, relationWorking12, LeftRelationMerge232323.deltas,
    LeftRelationMerge232323.source]
  decide +kernel
theorem relationClaim13 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking13 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232323)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨51142⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw13)
    (accumulator := relationWorking12) (working := relationWorking13)
    (reconstruction := relationReconstruction13)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt13 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim12 selector selectorLower selectorUpper witness
  · exact relationAgreement13
  · decide +kernel
theorem relationApplicationAt14 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232326 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking14 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw14 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase14 : Polynomial Owner :=
  subtract relationWorking13
    [{ coefficient := (-1), key := LeftRelationMerge232326.source }]
def relationReconstruction14 :
    MergeReconstructionAt history LeftRelationMerge232326.frameStart
      LeftRelationMerge232326.owner (.relation 232326) relationBase14
      relationWorking14 :=
  { deltas := LeftRelationMerge232326.deltas
    rows := LeftRelationMerge232326.rows
    agreement := by decide +kernel }
theorem relationAgreement14 :
    CanonicalAgreement (add relationBase14 relationReconstruction14.deltas)
      (relationPoly relationWorking13 LeftRelationMerge232326.source
        (relationContext LeftRelationMerge232326.source
          LeftRelationMerge232326.source.centralFactors 0 2) (-1)
        (relationRhsRaw14.map Term.toExact)) := by
  dsimp [relationReconstruction14, relationBase14, relationWorking14,
    relationRhsRaw14, relationWorking13, LeftRelationMerge232326.deltas,
    LeftRelationMerge232326.source]
  decide +kernel
theorem relationClaim14 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking14 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232326)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨32087⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw14)
    (accumulator := relationWorking13) (working := relationWorking14)
    (reconstruction := relationReconstruction14)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt14 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim13 selector selectorLower selectorUpper witness
  · exact relationAgreement14
  · decide +kernel
theorem relationApplicationAt15 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232329 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking15 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw15 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase15 : Polynomial Owner :=
  subtract relationWorking14
    [{ coefficient := (-1), key := LeftRelationMerge232329.source }]
def relationReconstruction15 :
    MergeReconstructionAt history LeftRelationMerge232329.frameStart
      LeftRelationMerge232329.owner (.relation 232329) relationBase15
      relationWorking15 :=
  { deltas := LeftRelationMerge232329.deltas
    rows := LeftRelationMerge232329.rows
    agreement := by decide +kernel }
theorem relationAgreement15 :
    CanonicalAgreement (add relationBase15 relationReconstruction15.deltas)
      (relationPoly relationWorking14 LeftRelationMerge232329.source
        (relationContext LeftRelationMerge232329.source
          LeftRelationMerge232329.source.centralFactors 0 2) (-1)
        (relationRhsRaw15.map Term.toExact)) := by
  dsimp [relationReconstruction15, relationBase15, relationWorking15,
    relationRhsRaw15, relationWorking14, LeftRelationMerge232329.deltas,
    LeftRelationMerge232329.source]
  decide +kernel
theorem relationClaim15 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking15 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232329)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨22067⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw15)
    (accumulator := relationWorking14) (working := relationWorking15)
    (reconstruction := relationReconstruction15)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt15 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim14 selector selectorLower selectorUpper witness
  · exact relationAgreement15
  · decide +kernel
theorem relationApplicationAt16 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232332 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking16 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw16 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase16 : Polynomial Owner :=
  subtract relationWorking15
    [{ coefficient := (-1), key := LeftRelationMerge232332.source }]
def relationReconstruction16 :
    MergeReconstructionAt history LeftRelationMerge232332.frameStart
      LeftRelationMerge232332.owner (.relation 232332) relationBase16
      relationWorking16 :=
  { deltas := LeftRelationMerge232332.deltas
    rows := LeftRelationMerge232332.rows
    agreement := by decide +kernel }
theorem relationAgreement16 :
    CanonicalAgreement (add relationBase16 relationReconstruction16.deltas)
      (relationPoly relationWorking15 LeftRelationMerge232332.source
        (relationContext LeftRelationMerge232332.source
          LeftRelationMerge232332.source.centralFactors 0 2) (-1)
        (relationRhsRaw16.map Term.toExact)) := by
  dsimp [relationReconstruction16, relationBase16, relationWorking16,
    relationRhsRaw16, relationWorking15, LeftRelationMerge232332.deltas,
    LeftRelationMerge232332.source]
  decide +kernel
theorem relationClaim16 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking16 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232332)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨18847⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw16)
    (accumulator := relationWorking15) (working := relationWorking16)
    (reconstruction := relationReconstruction16)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt16 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim15 selector selectorLower selectorUpper witness
  · exact relationAgreement16
  · decide +kernel
theorem relationApplicationAt17 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 232335 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking17 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18847⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32087⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34950⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37630⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40306⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45670⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51142⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54122⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57102⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63062⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66531⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationRhsRaw17 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68824⟩⟩] } }]
def relationBase17 : Polynomial Owner :=
  subtract relationWorking16
    [{ coefficient := (-1), key := LeftRelationMerge232335.source }]
def relationReconstruction17 :
    MergeReconstructionAt history LeftRelationMerge232335.frameStart
      LeftRelationMerge232335.owner (.relation 232335) relationBase17
      relationWorking17 :=
  { deltas := LeftRelationMerge232335.deltas
    rows := LeftRelationMerge232335.rows
    agreement := by decide +kernel }
theorem relationAgreement17 :
    CanonicalAgreement (add relationBase17 relationReconstruction17.deltas)
      (relationPoly relationWorking16 LeftRelationMerge232335.source
        (relationContext LeftRelationMerge232335.source
          LeftRelationMerge232335.source.centralFactors 0 2) (-1)
        (relationRhsRaw17.map Term.toExact)) := by
  dsimp [relationReconstruction17, relationBase17, relationWorking17,
    relationRhsRaw17, relationWorking16, LeftRelationMerge232335.deltas,
    LeftRelationMerge232335.source]
  decide +kernel
theorem relationClaim17 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking17 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 232335)
    (frameStart := 231586) (owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨16019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw17)
    (accumulator := relationWorking16) (working := relationWorking17)
    (reconstruction := relationReconstruction17)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt17 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim16 selector selectorLower selectorUpper witness
  · exact relationAgreement17
  · decide +kernel
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim17 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult232337

namespace SemanticResult232340
def owner : Owner := ⟨.program ⟨257⟩, ⟨67437⟩⟩
def rawTerms : List Term := Proof.Events907.exact232340RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232340
def producerEvent : Nat := 232339
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232340.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 231586, .finite 18, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult232340

namespace SemanticResult232345
def owner : Owner := ⟨.program ⟨257⟩, ⟨67439⟩⟩
def rawTerms : List Term := Proof.Events907.exact232345RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232345
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232345.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge232344.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge232344.frameStart)
    (transferEvent := 232343) (owner := owner)
    (leftResult := 232113) (rightResult := 232340)
    (working := LeftOperatorMerge232344.working)
    (reconstruction := LeftOperatorMerge232344.reconstruction)
    (leftReference := .predecessor 0 232341 .coefficient) (rightReference := .predecessor 1 232342 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult232113.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232340.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge232344.operationAgreement
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
end SemanticResult232345

namespace SemanticResult232348
def owner : Owner := ⟨.program ⟨257⟩, ⟨7233⟩⟩
def rawTerms : List Term := Proof.Events907.exact232348RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232348
def producerEvent : Nat := 232347
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232348.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 231586, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult232348

namespace SemanticResult232352
def owner : Owner := ⟨.program ⟨257⟩, ⟨67444⟩⟩
def rawTerms : List Term := Proof.Events907.exact232352RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232352
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232352.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232349) (rightBinding := 232350)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7233⟩) (rightExpression := ⟨67439⟩)
    (transferEvent := 232351)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232348.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232345.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232352

namespace SemanticResult232356
def owner : Owner := ⟨.program ⟨257⟩, ⟨71209⟩⟩
def rawTerms : List Term := Proof.Events907.exact232356RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 232356
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult232356.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 232353) (rightBinding := 232354)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67444⟩) (rightExpression := ⟨71205⟩)
    (transferEvent := 232355)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult232352.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult232337.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult232356

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
