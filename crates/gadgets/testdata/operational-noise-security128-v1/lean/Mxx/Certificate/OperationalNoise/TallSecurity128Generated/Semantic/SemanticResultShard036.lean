import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard036
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard033
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard034
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard035

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult4476
def owner : Owner := ⟨.program ⟨257⟩, ⟨18958⟩⟩
def rawTerms : List Term := Proof.Events017.exact4476RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4476
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4476.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4473) (rightBinding := 4474)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16112⟩) (rightExpression := ⟨18957⟩)
    (transferEvent := 4475)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4472.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4460.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4476

namespace SemanticResult4480
def owner : Owner := ⟨.program ⟨257⟩, ⟨22178⟩⟩
def rawTerms : List Term := Proof.Events017.exact4480RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4480.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4477) (rightBinding := 4478)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18958⟩) (rightExpression := ⟨22177⟩)
    (transferEvent := 4479)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4476.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4452.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4480

namespace SemanticResult4484
def owner : Owner := ⟨.program ⟨257⟩, ⟨32198⟩⟩
def rawTerms : List Term := Proof.Events017.exact4484RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4484
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4484.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4481) (rightBinding := 4482)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22178⟩) (rightExpression := ⟨32197⟩)
    (transferEvent := 4483)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4480.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4444.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4484

namespace SemanticResult4488
def owner : Owner := ⟨.program ⟨257⟩, ⟨51262⟩⟩
def rawTerms : List Term := Proof.Events017.exact4488RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4488.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4485) (rightBinding := 4486)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32198⟩) (rightExpression := ⟨51261⟩)
    (transferEvent := 4487)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4484.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4436.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4488

namespace SemanticResult4492
def owner : Owner := ⟨.program ⟨257⟩, ⟨54242⟩⟩
def rawTerms : List Term := Proof.Events017.exact4492RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4492
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4492.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4489) (rightBinding := 4490)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51262⟩) (rightExpression := ⟨54241⟩)
    (transferEvent := 4491)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4488.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4428.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4492

namespace SemanticResult4496
def owner : Owner := ⟨.program ⟨257⟩, ⟨57222⟩⟩
def rawTerms : List Term := Proof.Events017.exact4496RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4496.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4493) (rightBinding := 4494)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54242⟩) (rightExpression := ⟨57221⟩)
    (transferEvent := 4495)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4492.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4420.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4496

namespace SemanticResult4500
def owner : Owner := ⟨.program ⟨257⟩, ⟨60202⟩⟩
def rawTerms : List Term := Proof.Events017.exact4500RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4500.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4497) (rightBinding := 4498)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57222⟩) (rightExpression := ⟨60201⟩)
    (transferEvent := 4499)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4496.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4412.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4500

namespace SemanticResult4504
def owner : Owner := ⟨.program ⟨257⟩, ⟨63182⟩⟩
def rawTerms : List Term := Proof.Events017.exact4504RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4504
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4504.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4501) (rightBinding := 4502)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60202⟩) (rightExpression := ⟨63181⟩)
    (transferEvent := 4503)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4500.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4404.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4504

namespace SemanticResult4508
def owner : Owner := ⟨.program ⟨257⟩, ⟨66940⟩⟩
def rawTerms : List Term := Proof.Events017.exact4508RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4508
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4508.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4505) (rightBinding := 4506)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63182⟩) (rightExpression := ⟨66939⟩)
    (transferEvent := 4507)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4504.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4396.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4508

namespace SemanticResult4512
def owner : Owner := ⟨.program ⟨257⟩, ⟨66941⟩⟩
def rawTerms : List Term := Proof.Events017.exact4512RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4512
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4512.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4509) (rightBinding := 4510)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66940⟩) (rightExpression := ⟨26688⟩)
    (transferEvent := 4511)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4508.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4388.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4512

namespace SemanticResult4516
def owner : Owner := ⟨.program ⟨257⟩, ⟨66942⟩⟩
def rawTerms : List Term := Proof.Events017.exact4516RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4516.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4513) (rightBinding := 4514)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66941⟩) (rightExpression := ⟨29368⟩)
    (transferEvent := 4515)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4380.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4516

namespace SemanticResult4520
def owner : Owner := ⟨.program ⟨257⟩, ⟨66943⟩⟩
def rawTerms : List Term := Proof.Events017.exact4520RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4520.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4517) (rightBinding := 4518)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66942⟩) (rightExpression := ⟨35025⟩)
    (transferEvent := 4519)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4516.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4372.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4520

namespace SemanticResult4524
def owner : Owner := ⟨.program ⟨257⟩, ⟨66944⟩⟩
def rawTerms : List Term := Proof.Events017.exact4524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4524.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4521) (rightBinding := 4522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66943⟩) (rightExpression := ⟨37705⟩)
    (transferEvent := 4523)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4364.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4524

namespace SemanticResult4528
def owner : Owner := ⟨.program ⟨257⟩, ⟨66945⟩⟩
def rawTerms : List Term := Proof.Events017.exact4528RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4528.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4525) (rightBinding := 4526)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66944⟩) (rightExpression := ⟨40388⟩)
    (transferEvent := 4527)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4524.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4356.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4528

namespace SemanticResult4532
def owner : Owner := ⟨.program ⟨257⟩, ⟨66946⟩⟩
def rawTerms : List Term := Proof.Events017.exact4532RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4532
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4532.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4529) (rightBinding := 4530)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66945⟩) (rightExpression := ⟨43068⟩)
    (transferEvent := 4531)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4528.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4348.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4532

namespace SemanticResult4536
def owner : Owner := ⟨.program ⟨257⟩, ⟨66947⟩⟩
def rawTerms : List Term := Proof.Events017.exact4536RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 4536
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult4536.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 4533) (rightBinding := 4534)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨66946⟩) (rightExpression := ⟨45745⟩)
    (transferEvent := 4535)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult4532.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult4340.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult4536

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
