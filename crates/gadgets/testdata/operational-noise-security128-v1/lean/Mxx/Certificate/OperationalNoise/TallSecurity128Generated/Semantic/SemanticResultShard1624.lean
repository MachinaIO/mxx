import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1624
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1559
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1602
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1606
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1609
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1613
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1617
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1620
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1623

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult230773
def owner : Owner := ⟨.program ⟨257⟩, ⟨7179⟩⟩
def rawTerms : List Term := Proof.Events901.exact230773RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 230773
def producerEvent : Nat := 230772
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230773.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 230703, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult230773

namespace SemanticResult230777
def owner : Owner := ⟨.program ⟨257⟩, ⟨17205⟩⟩
def rawTerms : List Term := Proof.Events901.exact230777RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 230777
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230777.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 230774) (rightBinding := 230775)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7179⟩) (rightExpression := ⟨17204⟩)
    (transferEvent := 230776)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230773.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult230770.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult230777

namespace SemanticResult230785
def owner : Owner := ⟨.program ⟨257⟩, ⟨17734⟩⟩
def rawTerms : List Term := Proof.Events901.exact230785RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 230785
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230785.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge230781.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge230781.frameStart)
    (transferEvent := 230780) (owner := owner)
    (leftResult := 230777) (rightResult := 230754)
    (working := LeftOperatorMerge230781.working)
    (reconstruction := LeftOperatorMerge230781.reconstruction)
    (leftReference := .predecessor 0 230778 .coefficient) (rightReference := .predecessor 1 230779 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult230777.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult230754.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge230781.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 230783 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16992⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16992⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge230781.working
    [{ coefficient := (-1), key := LeftRelationMerge230783.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge230783.frameStart
      LeftRelationMerge230783.owner (.relation 230783) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge230783.deltas
    rows := LeftRelationMerge230783.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge230781.working LeftRelationMerge230783.source
        (relationContext LeftRelationMerge230783.source
          LeftRelationMerge230783.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge230781.working, LeftRelationMerge230783.deltas,
    LeftRelationMerge230783.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 230783)
    (frameStart := 230703) (owner := ⟨.program ⟨257⟩, ⟨17734⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨15780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge230781.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact mergeClaim selector selectorLower selectorUpper witness
  · exact relationAgreement0
  · decide +kernel
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult230785

namespace SemanticResult230788
def owner : Owner := ⟨.program ⟨257⟩, ⟨16019⟩⟩
def rawTerms : List Term := Proof.Events901.exact230788RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 230788
def producerEvent : Nat := 230787
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230788.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 230703, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult230788

namespace SemanticResult230793
def owner : Owner := ⟨.program ⟨257⟩, ⟨16020⟩⟩
def rawTerms : List Term := Proof.Events901.exact230793RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 230793
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230793.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge230792.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge230792.frameStart)
    (transferEvent := 230791) (owner := owner)
    (leftResult := 230765) (rightResult := 230788)
    (working := LeftOperatorMerge230792.working)
    (reconstruction := LeftOperatorMerge230792.reconstruction)
    (leftReference := .predecessor 0 230789 .coefficient) (rightReference := .predecessor 1 230790 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult230765.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult230788.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge230792.operationAgreement
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
end SemanticResult230793

namespace SemanticResult230796
def owner : Owner := ⟨.program ⟨257⟩, ⟨7198⟩⟩
def rawTerms : List Term := Proof.Events901.exact230796RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 230796
def producerEvent : Nat := 230795
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230796.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 230703, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult230796

namespace SemanticResult230800
def owner : Owner := ⟨.program ⟨257⟩, ⟨16021⟩⟩
def rawTerms : List Term := Proof.Events901.exact230800RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 230800
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230800.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 230797) (rightBinding := 230798)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7198⟩) (rightExpression := ⟨16020⟩)
    (transferEvent := 230799)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230796.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult230793.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult230800

namespace SemanticResult230804
def owner : Owner := ⟨.program ⟨257⟩, ⟨17737⟩⟩
def rawTerms : List Term := Proof.Events901.exact230804RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 230804
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230804.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 230801) (rightBinding := 230802)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16021⟩) (rightExpression := ⟨17734⟩)
    (transferEvent := 230803)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230800.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult230785.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult230804

namespace SemanticResult230813
def owner : Owner := ⟨.program ⟨257⟩, ⟨16579⟩⟩
def rawTerms : List Term := Proof.Events901.exact230813RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 230813
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230813.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge230648.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge230648.frameStart)
    (owner := owner) (leftOwner := SemanticResult222245.owner)
    (rightOwner := SemanticResult230642.owner)
    (leftResult := 222245) (rightResult := 230642)
    (leftActual := SemanticResult222245.actual selector witness)
    (rightActual := SemanticResult230642.actual selector witness)
    (leftRaw := SemanticResult222245.rawTerms)
    (rightRaw := SemanticResult230642.rawTerms)
    (working := LeftOperatorMerge230648.working)
    (leftBinding := 230643) (rightBinding := 230644)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5581⟩) (rightExpression := ⟨16578⟩)
    (coefficientTransfer := 230645) (summaryTransfer := 230647)
    (rightCoefficientProducer := 230641)
    (rightSummaryTransfer := 230646)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge230648.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound230641.actual selector witness)
    (summaryMagnitude := LeftBound230647.actual selector witness)
    (reconstruction := LeftOperatorMerge230648.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult222245.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult230642.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound230641.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound230641.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge230648.operationAgreement
  · exact LeftBound230647.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge230648.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 230808 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16992⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17733⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16992⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge230648.working
    [{ coefficient := (1), key := LeftRelationMerge230808.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge230808.frameStart
      LeftRelationMerge230808.owner (.relation 230808) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge230808.deltas
    rows := LeftRelationMerge230808.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge230648.working LeftRelationMerge230808.source
        (relationContext LeftRelationMerge230808.source
          LeftRelationMerge230808.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge230648.working, LeftRelationMerge230808.deltas,
    LeftRelationMerge230808.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 230808)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16579⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16576⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge230648.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
  · exact mergeClaim selector selectorLower selectorUpper witness
  · exact relationAgreement0
  · decide +kernel
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult230813

namespace SemanticResult230820
def owner : Owner := ⟨.program ⟨257⟩, ⟨17736⟩⟩
def rawTerms : List Term := Proof.Events901.exact230820RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 230820
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230820.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge230817.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult230813.owner)
    (rightOwner := SemanticResult230635.owner)
    (leftResult := 230813) (rightResult := 230635)
    (leftActual := SemanticResult230813.actual selector witness)
    (rightActual := SemanticResult230635.actual selector witness)
    (leftRaw := SemanticResult230813.rawTerms)
    (rightRaw := SemanticResult230635.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 230814) (rightBinding := 230815)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16579⟩) (rightExpression := ⟨17735⟩)
    (coefficientTransfer := 230816) (summaryTransfer := 230819)
    (base := LeftOperatorMerge230817.base)
    (reconstruction := LeftOperatorMerge230817.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230813.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult230635.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge230817.operationAgreement
  · rfl
  · decide
end SemanticResult230820

namespace SemanticResult230825
def owner : Owner := ⟨.program ⟨257⟩, ⟨20625⟩⟩
def rawTerms : List Term := Proof.Events901.exact230825RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 230825
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230825.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult230820.owner)
    (rightOwner := SemanticResult230338.owner)
    (leftResult := 230820) (rightResult := 230338)
    (leftActual := SemanticResult230820.actual selector witness)
    (rightActual := SemanticResult230338.actual selector witness)
    (leftRaw := SemanticResult230820.rawTerms)
    (rightRaw := SemanticResult230338.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 230821) (rightBinding := 230822)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17736⟩) (rightExpression := ⟨20624⟩)
    (transferEvent := 230823) (summaryTransferEvent := 230824)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230820.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult230338.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult230825

namespace SemanticResult230830
def owner : Owner := ⟨.program ⟨257⟩, ⟨23845⟩⟩
def rawTerms : List Term := Proof.Events901.exact230830RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 230830
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230830.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult230825.owner)
    (rightOwner := SemanticResult229856.owner)
    (leftResult := 230825) (rightResult := 229856)
    (leftActual := SemanticResult230825.actual selector witness)
    (rightActual := SemanticResult229856.actual selector witness)
    (leftRaw := SemanticResult230825.rawTerms)
    (rightRaw := SemanticResult229856.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 230826) (rightBinding := 230827)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20625⟩) (rightExpression := ⟨23844⟩)
    (transferEvent := 230828) (summaryTransferEvent := 230829)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230825.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult229856.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult230830

namespace SemanticResult230835
def owner : Owner := ⟨.program ⟨257⟩, ⟨33865⟩⟩
def rawTerms : List Term := Proof.Events901.exact230835RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 230835
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230835.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult230830.owner)
    (rightOwner := SemanticResult229374.owner)
    (leftResult := 230830) (rightResult := 229374)
    (leftActual := SemanticResult230830.actual selector witness)
    (rightActual := SemanticResult229374.actual selector witness)
    (leftRaw := SemanticResult230830.rawTerms)
    (rightRaw := SemanticResult229374.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 230831) (rightBinding := 230832)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23845⟩) (rightExpression := ⟨33864⟩)
    (transferEvent := 230833) (summaryTransferEvent := 230834)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230830.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult229374.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult230835

namespace SemanticResult230840
def owner : Owner := ⟨.program ⟨257⟩, ⟨52925⟩⟩
def rawTerms : List Term := Proof.Events901.exact230840RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 230840
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230840.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult230835.owner)
    (rightOwner := SemanticResult228892.owner)
    (leftResult := 230835) (rightResult := 228892)
    (leftActual := SemanticResult230835.actual selector witness)
    (rightActual := SemanticResult228892.actual selector witness)
    (leftRaw := SemanticResult230835.rawTerms)
    (rightRaw := SemanticResult228892.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 230836) (rightBinding := 230837)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33865⟩) (rightExpression := ⟨52924⟩)
    (transferEvent := 230838) (summaryTransferEvent := 230839)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230835.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult228892.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult230840

namespace SemanticResult230845
def owner : Owner := ⟨.program ⟨257⟩, ⟨55905⟩⟩
def rawTerms : List Term := Proof.Events901.exact230845RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 230845
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230845.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult230840.owner)
    (rightOwner := SemanticResult228410.owner)
    (leftResult := 230840) (rightResult := 228410)
    (leftActual := SemanticResult230840.actual selector witness)
    (rightActual := SemanticResult228410.actual selector witness)
    (leftRaw := SemanticResult230840.rawTerms)
    (rightRaw := SemanticResult228410.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 230841) (rightBinding := 230842)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52925⟩) (rightExpression := ⟨55904⟩)
    (transferEvent := 230843) (summaryTransferEvent := 230844)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230840.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult228410.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult230845

namespace SemanticResult230850
def owner : Owner := ⟨.program ⟨257⟩, ⟨58885⟩⟩
def rawTerms : List Term := Proof.Events901.exact230850RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 230850
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult230850.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult230845.owner)
    (rightOwner := SemanticResult227928.owner)
    (leftResult := 230845) (rightResult := 227928)
    (leftActual := SemanticResult230845.actual selector witness)
    (rightActual := SemanticResult227928.actual selector witness)
    (leftRaw := SemanticResult230845.rawTerms)
    (rightRaw := SemanticResult227928.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 230846) (rightBinding := 230847)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55905⟩) (rightExpression := ⟨58884⟩)
    (transferEvent := 230848) (summaryTransferEvent := 230849)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult230845.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult227928.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult230850

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
