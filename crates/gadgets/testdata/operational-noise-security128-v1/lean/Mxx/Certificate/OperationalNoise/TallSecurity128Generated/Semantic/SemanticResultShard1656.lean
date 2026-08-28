import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1656
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1557
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1640
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1642
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1643
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1644
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1646
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1647
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1649
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1650
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1651
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1653
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1654
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1655

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult236276
def owner : Owner := ⟨.program ⟨257⟩, ⟨8484⟩⟩
def rawTerms : List Term := Proof.Events922.exact236276RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 236276
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236276.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge236275.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge236275.frameStart)
    (transferEvent := 236274) (owner := owner)
    (leftResult := 222023) (rightResult := 15896)
    (working := LeftOperatorMerge236275.working)
    (reconstruction := LeftOperatorMerge236275.reconstruction)
    (leftReference := .predecessor 0 236272 .coefficient) (rightReference := .predecessor 1 236273 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult222023.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15896.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge236275.operationAgreement
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
end SemanticResult236276

namespace SemanticResult236280
def owner : Owner := ⟨.program ⟨257⟩, ⟨9377⟩⟩
def rawTerms : List Term := Proof.Events922.exact236280RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 236280
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236280.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 236277) (rightBinding := 236278)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8484⟩) (rightExpression := ⟨7082⟩)
    (transferEvent := 236279)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236276.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult236271.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236280

namespace SemanticResult236286
def owner : Owner := ⟨.program ⟨257⟩, ⟨9378⟩⟩
def rawTerms : List Term := Proof.Events922.exact236286RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 236286
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236286.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 236283) (survivorTransfer := 236284)
    (survivorEvent := 236285) (resultEvent := resultEvent)
    (rightCoefficientProducer := 31515)
    (owner := owner) (leftOwner := SemanticResult236280.owner)
    (rightOwner := SemanticResult31516.owner)
    (leftResult := 236280) (rightResult := 31516)
    (leftBinding := 236281) (rightBinding := 236282)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9377⟩) (rightExpression := ⟨118⟩)
    (leftActual := SemanticResult236280.actual selector witness)
    (rightActual := SemanticResult31516.actual selector witness)
    (leftRaw := SemanticResult236280.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound31515.actual selector witness)
    (survivorMagnitude := LeftBound236284.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236280.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)
  · exact LeftBound236284.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult236286

namespace SemanticResult236293
def owner : Owner := ⟨.program ⟨257⟩, ⟨9474⟩⟩
def rawTerms : List Term := Proof.Events923.exact236293RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 236293
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236293.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge236290.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236286.owner)
    (rightOwner := SemanticResult236286.owner)
    (leftResult := 236286) (rightResult := 236286)
    (leftActual := SemanticResult236286.actual selector witness)
    (rightActual := SemanticResult236286.actual selector witness)
    (leftRaw := SemanticResult236286.rawTerms)
    (rightRaw := SemanticResult236286.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236287) (rightBinding := 236288)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9378⟩) (rightExpression := ⟨9378⟩)
    (coefficientTransfer := 236289) (summaryTransfer := 236292)
    (base := LeftOperatorMerge236290.base)
    (reconstruction := LeftOperatorMerge236290.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236286.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult236286.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge236290.operationAgreement
  · rfl
  · decide
end SemanticResult236293

namespace SemanticResult236298
def owner : Owner := ⟨.program ⟨257⟩, ⟨17731⟩⟩
def rawTerms : List Term := Proof.Events923.exact236298RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 236298
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236298.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236293.owner)
    (rightOwner := SemanticResult236266.owner)
    (leftResult := 236293) (rightResult := 236266)
    (leftActual := SemanticResult236293.actual selector witness)
    (rightActual := SemanticResult236266.actual selector witness)
    (leftRaw := SemanticResult236293.rawTerms)
    (rightRaw := SemanticResult236266.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236294) (rightBinding := 236295)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9474⟩) (rightExpression := ⟨17730⟩)
    (transferEvent := 236296) (summaryTransferEvent := 236297)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236293.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult236266.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236298

namespace SemanticResult236303
def owner : Owner := ⟨.program ⟨257⟩, ⟨20619⟩⟩
def rawTerms : List Term := Proof.Events923.exact236303RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 236303
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236303.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236298.owner)
    (rightOwner := SemanticResult236054.owner)
    (leftResult := 236298) (rightResult := 236054)
    (leftActual := SemanticResult236298.actual selector witness)
    (rightActual := SemanticResult236054.actual selector witness)
    (leftRaw := SemanticResult236298.rawTerms)
    (rightRaw := SemanticResult236054.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236299) (rightBinding := 236300)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17731⟩) (rightExpression := ⟨20618⟩)
    (transferEvent := 236301) (summaryTransferEvent := 236302)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236298.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult236054.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236303

namespace SemanticResult236308
def owner : Owner := ⟨.program ⟨257⟩, ⟨23839⟩⟩
def rawTerms : List Term := Proof.Events923.exact236308RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 236308
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236308.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236303.owner)
    (rightOwner := SemanticResult235842.owner)
    (leftResult := 236303) (rightResult := 235842)
    (leftActual := SemanticResult236303.actual selector witness)
    (rightActual := SemanticResult235842.actual selector witness)
    (leftRaw := SemanticResult236303.rawTerms)
    (rightRaw := SemanticResult235842.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236304) (rightBinding := 236305)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20619⟩) (rightExpression := ⟨23838⟩)
    (transferEvent := 236306) (summaryTransferEvent := 236307)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236303.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult235842.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236308

namespace SemanticResult236313
def owner : Owner := ⟨.program ⟨257⟩, ⟨33859⟩⟩
def rawTerms : List Term := Proof.Events923.exact236313RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 236313
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236313.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236308.owner)
    (rightOwner := SemanticResult235630.owner)
    (leftResult := 236308) (rightResult := 235630)
    (leftActual := SemanticResult236308.actual selector witness)
    (rightActual := SemanticResult235630.actual selector witness)
    (leftRaw := SemanticResult236308.rawTerms)
    (rightRaw := SemanticResult235630.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236309) (rightBinding := 236310)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23839⟩) (rightExpression := ⟨33858⟩)
    (transferEvent := 236311) (summaryTransferEvent := 236312)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236308.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult235630.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236313

namespace SemanticResult236318
def owner : Owner := ⟨.program ⟨257⟩, ⟨52919⟩⟩
def rawTerms : List Term := Proof.Events923.exact236318RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 236318
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236318.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236313.owner)
    (rightOwner := SemanticResult235418.owner)
    (leftResult := 236313) (rightResult := 235418)
    (leftActual := SemanticResult236313.actual selector witness)
    (rightActual := SemanticResult235418.actual selector witness)
    (leftRaw := SemanticResult236313.rawTerms)
    (rightRaw := SemanticResult235418.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236314) (rightBinding := 236315)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33859⟩) (rightExpression := ⟨52918⟩)
    (transferEvent := 236316) (summaryTransferEvent := 236317)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236313.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult235418.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236318

namespace SemanticResult236323
def owner : Owner := ⟨.program ⟨257⟩, ⟨55899⟩⟩
def rawTerms : List Term := Proof.Events923.exact236323RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 236323
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236323.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236318.owner)
    (rightOwner := SemanticResult235206.owner)
    (leftResult := 236318) (rightResult := 235206)
    (leftActual := SemanticResult236318.actual selector witness)
    (rightActual := SemanticResult235206.actual selector witness)
    (leftRaw := SemanticResult236318.rawTerms)
    (rightRaw := SemanticResult235206.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236319) (rightBinding := 236320)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52919⟩) (rightExpression := ⟨55898⟩)
    (transferEvent := 236321) (summaryTransferEvent := 236322)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236318.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult235206.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236323

namespace SemanticResult236328
def owner : Owner := ⟨.program ⟨257⟩, ⟨58879⟩⟩
def rawTerms : List Term := Proof.Events923.exact236328RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 236328
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236328.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236323.owner)
    (rightOwner := SemanticResult234994.owner)
    (leftResult := 236323) (rightResult := 234994)
    (leftActual := SemanticResult236323.actual selector witness)
    (rightActual := SemanticResult234994.actual selector witness)
    (leftRaw := SemanticResult236323.rawTerms)
    (rightRaw := SemanticResult234994.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236324) (rightBinding := 236325)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55899⟩) (rightExpression := ⟨58878⟩)
    (transferEvent := 236326) (summaryTransferEvent := 236327)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236323.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult234994.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236328

namespace SemanticResult236333
def owner : Owner := ⟨.program ⟨257⟩, ⟨61859⟩⟩
def rawTerms : List Term := Proof.Events923.exact236333RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 236333
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236333.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236328.owner)
    (rightOwner := SemanticResult234782.owner)
    (leftResult := 236328) (rightResult := 234782)
    (leftActual := SemanticResult236328.actual selector witness)
    (rightActual := SemanticResult234782.actual selector witness)
    (leftRaw := SemanticResult236328.rawTerms)
    (rightRaw := SemanticResult234782.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236329) (rightBinding := 236330)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58879⟩) (rightExpression := ⟨61858⟩)
    (transferEvent := 236331) (summaryTransferEvent := 236332)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236328.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult234782.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236333

namespace SemanticResult236338
def owner : Owner := ⟨.program ⟨257⟩, ⟨64839⟩⟩
def rawTerms : List Term := Proof.Events923.exact236338RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 236338
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236338.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236333.owner)
    (rightOwner := SemanticResult234570.owner)
    (leftResult := 236333) (rightResult := 234570)
    (leftActual := SemanticResult236333.actual selector witness)
    (rightActual := SemanticResult234570.actual selector witness)
    (leftRaw := SemanticResult236333.rawTerms)
    (rightRaw := SemanticResult234570.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236334) (rightBinding := 236335)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61859⟩) (rightExpression := ⟨64838⟩)
    (transferEvent := 236336) (summaryTransferEvent := 236337)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236333.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult234570.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236338

namespace SemanticResult236343
def owner : Owner := ⟨.program ⟨257⟩, ⟨70088⟩⟩
def rawTerms : List Term := Proof.Events923.exact236343RawTerms
def summary : Bound := (.finite 3456353380086899479155517117627148481331252)
def resultEvent : Nat := 236343
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236343.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236338.owner)
    (rightOwner := SemanticResult234358.owner)
    (leftResult := 236338) (rightResult := 234358)
    (leftActual := SemanticResult236338.actual selector witness)
    (rightActual := SemanticResult234358.actual selector witness)
    (leftRaw := SemanticResult236338.rawTerms)
    (rightRaw := SemanticResult234358.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3110701272581949232038858886277070355169332)
    (rightMaximum := 345652107504950247116658231350078126161920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236339) (rightBinding := 236340)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64839⟩) (rightExpression := ⟨70087⟩)
    (transferEvent := 236341) (summaryTransferEvent := 236342)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236338.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult234358.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236343

namespace SemanticResult236348
def owner : Owner := ⟨.program ⟨257⟩, ⟨70089⟩⟩
def rawTerms : List Term := Proof.Events923.exact236348RawTerms
def summary : Bound := (.finite 3802007596962448506045899439491360353157172)
def resultEvent : Nat := 236348
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236348.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236343.owner)
    (rightOwner := SemanticResult234146.owner)
    (leftResult := 236343) (rightResult := 234146)
    (leftActual := SemanticResult236343.actual selector witness)
    (rightActual := SemanticResult234146.actual selector witness)
    (leftRaw := SemanticResult236343.rawTerms)
    (rightRaw := SemanticResult234146.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3456353380086899479155517117627148481331252)
    (rightMaximum := 345654216875549026890382321864211871825920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236344) (rightBinding := 236345)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70088⟩) (rightExpression := ⟨28262⟩)
    (transferEvent := 236346) (summaryTransferEvent := 236347)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236343.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult234146.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236348

namespace SemanticResult236353
def owner : Owner := ⟨.program ⟨257⟩, ⟨70090⟩⟩
def rawTerms : List Term := Proof.Events923.exact236353RawTerms
def summary : Bound := (.finite 4147668141949793872257454032897973461975092)
def resultEvent : Nat := 236353
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult236353.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult236348.owner)
    (rightOwner := SemanticResult233934.owner)
    (leftResult := 236348) (rightResult := 233934)
    (leftActual := SemanticResult236348.actual selector witness)
    (rightActual := SemanticResult233934.actual selector witness)
    (leftRaw := SemanticResult236348.rawTerms)
    (rightRaw := SemanticResult233934.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3802007596962448506045899439491360353157172)
    (rightMaximum := 345660544987345366211554593406613108817920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 236349) (rightBinding := 236350)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨70089⟩) (rightExpression := ⟨30942⟩)
    (transferEvent := 236351) (summaryTransferEvent := 236352)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult236348.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult233934.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult236353

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
