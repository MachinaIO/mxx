import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard012
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard005
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard010
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard011

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult1468
def owner : Owner := ⟨.program ⟨257⟩, ⟨19033⟩⟩
def rawTerms : List Term := Proof.Events005.exact1468RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1468
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1468.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge1467.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge1467.frameStart)
    (transferEvent := 1466) (owner := owner)
    (leftResult := 1463) (rightResult := 703)
    (working := LeftOperatorMerge1467.working)
    (reconstruction := LeftOperatorMerge1467.reconstruction)
    (leftReference := .predecessor 0 1464 .coefficient) (rightReference := .predecessor 1 1465 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult1463.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult703.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge1467.operationAgreement
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
end SemanticResult1468

namespace SemanticResult1471
def owner : Owner := ⟨.program ⟨257⟩, ⟨16174⟩⟩
def rawTerms : List Term := Proof.Events005.exact1471RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1471
def producerEvent : Nat := 1470
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1471.actual selector witness
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
end SemanticResult1471

namespace SemanticResult1476
def owner : Owner := ⟨.program ⟨257⟩, ⟨16175⟩⟩
def rawTerms : List Term := Proof.Events005.exact1476RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1476
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1476.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge1475.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge1475.frameStart)
    (transferEvent := 1474) (owner := owner)
    (leftResult := 1471) (rightResult := 713)
    (working := LeftOperatorMerge1475.working)
    (reconstruction := LeftOperatorMerge1475.reconstruction)
    (leftReference := .predecessor 0 1472 .coefficient) (rightReference := .predecessor 1 1473 .coefficient)
    (facts := ⟨true, true, none, some 1, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult1471.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult713.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge1475.operationAgreement
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
end SemanticResult1476

namespace SemanticResult1480
def owner : Owner := ⟨.program ⟨257⟩, ⟨16176⟩⟩
def rawTerms : List Term := Proof.Events005.exact1480RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1480
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1480.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1477) (rightBinding := 1478)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6728⟩) (rightExpression := ⟨16175⟩)
    (transferEvent := 1479)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult728.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1476.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1480

namespace SemanticResult1484
def owner : Owner := ⟨.program ⟨257⟩, ⟨19034⟩⟩
def rawTerms : List Term := Proof.Events005.exact1484RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1484
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1484.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1481) (rightBinding := 1482)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16176⟩) (rightExpression := ⟨19033⟩)
    (transferEvent := 1483)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1480.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1468.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1484

namespace SemanticResult1488
def owner : Owner := ⟨.program ⟨257⟩, ⟨22254⟩⟩
def rawTerms : List Term := Proof.Events005.exact1488RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1488
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1488.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1485) (rightBinding := 1486)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨19034⟩) (rightExpression := ⟨22253⟩)
    (transferEvent := 1487)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1484.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1460.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1488

namespace SemanticResult1492
def owner : Owner := ⟨.program ⟨257⟩, ⟨32274⟩⟩
def rawTerms : List Term := Proof.Events005.exact1492RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1492
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1492.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1489) (rightBinding := 1490)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22254⟩) (rightExpression := ⟨32273⟩)
    (transferEvent := 1491)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1488.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1452.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1492

namespace SemanticResult1496
def owner : Owner := ⟨.program ⟨257⟩, ⟨51338⟩⟩
def rawTerms : List Term := Proof.Events005.exact1496RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1496
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1496.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1493) (rightBinding := 1494)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨32274⟩) (rightExpression := ⟨51337⟩)
    (transferEvent := 1495)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1492.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1444.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1496

namespace SemanticResult1500
def owner : Owner := ⟨.program ⟨257⟩, ⟨54318⟩⟩
def rawTerms : List Term := Proof.Events005.exact1500RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1500
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1500.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1497) (rightBinding := 1498)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨51338⟩) (rightExpression := ⟨54317⟩)
    (transferEvent := 1499)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1496.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1436.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1500

namespace SemanticResult1504
def owner : Owner := ⟨.program ⟨257⟩, ⟨57298⟩⟩
def rawTerms : List Term := Proof.Events005.exact1504RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1504
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1504.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1501) (rightBinding := 1502)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨54318⟩) (rightExpression := ⟨57297⟩)
    (transferEvent := 1503)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1500.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1428.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1504

namespace SemanticResult1508
def owner : Owner := ⟨.program ⟨257⟩, ⟨60278⟩⟩
def rawTerms : List Term := Proof.Events005.exact1508RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1508
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1508.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1505) (rightBinding := 1506)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨57298⟩) (rightExpression := ⟨60277⟩)
    (transferEvent := 1507)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1504.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1420.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1508

namespace SemanticResult1512
def owner : Owner := ⟨.program ⟨257⟩, ⟨63258⟩⟩
def rawTerms : List Term := Proof.Events005.exact1512RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1512
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1512.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1509) (rightBinding := 1510)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨60278⟩) (rightExpression := ⟨63257⟩)
    (transferEvent := 1511)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1508.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1412.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1512

namespace SemanticResult1516
def owner : Owner := ⟨.program ⟨257⟩, ⟨67220⟩⟩
def rawTerms : List Term := Proof.Events005.exact1516RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1516
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1516.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1513) (rightBinding := 1514)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨63258⟩) (rightExpression := ⟨67219⟩)
    (transferEvent := 1515)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1512.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1404.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1516

namespace SemanticResult1520
def owner : Owner := ⟨.program ⟨257⟩, ⟨67221⟩⟩
def rawTerms : List Term := Proof.Events005.exact1520RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1520
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1520.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1517) (rightBinding := 1518)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67220⟩) (rightExpression := ⟨26740⟩)
    (transferEvent := 1519)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1516.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1396.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1520

namespace SemanticResult1524
def owner : Owner := ⟨.program ⟨257⟩, ⟨67222⟩⟩
def rawTerms : List Term := Proof.Events005.exact1524RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1524.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1521) (rightBinding := 1522)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67221⟩) (rightExpression := ⟨29420⟩)
    (transferEvent := 1523)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1520.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1388.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1524

namespace SemanticResult1528
def owner : Owner := ⟨.program ⟨257⟩, ⟨67223⟩⟩
def rawTerms : List Term := Proof.Events005.exact1528RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 1528
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult1528.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 1525) (rightBinding := 1526)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67222⟩) (rightExpression := ⟨35077⟩)
    (transferEvent := 1527)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult1524.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1380.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult1528

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
