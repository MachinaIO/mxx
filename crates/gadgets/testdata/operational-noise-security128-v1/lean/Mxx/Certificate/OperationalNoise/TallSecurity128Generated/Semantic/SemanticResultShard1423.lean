import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1423
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1358
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1393
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1397
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1401
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1405
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1408
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1412
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1416
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1419
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1422

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult201535
def owner : Owner := ⟨.program ⟨257⟩, ⟨17818⟩⟩
def rawTerms : List Term := Proof.Events787.exact201535RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 201535
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201535.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge201531.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge201531.frameStart)
    (transferEvent := 201530) (owner := owner)
    (leftResult := 201527) (rightResult := 201504)
    (working := LeftOperatorMerge201531.working)
    (reconstruction := LeftOperatorMerge201531.reconstruction)
    (leftReference := .predecessor 0 201528 .coefficient) (rightReference := .predecessor 1 201529 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult201527.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult201504.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge201531.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 201533 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17019⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17019⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge201531.working
    [{ coefficient := (-1), key := LeftRelationMerge201533.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge201533.frameStart
      LeftRelationMerge201533.owner (.relation 201533) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge201533.deltas
    rows := LeftRelationMerge201533.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge201531.working LeftRelationMerge201533.source
        (relationContext LeftRelationMerge201533.source
          LeftRelationMerge201533.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge201531.working, LeftRelationMerge201533.deltas,
    LeftRelationMerge201533.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 201533)
    (frameStart := 201453) (owner := ⟨.program ⟨257⟩, ⟨17818⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge201531.working) (working := relationWorking0)
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
end SemanticResult201535

namespace SemanticResult201538
def owner : Owner := ⟨.program ⟨257⟩, ⟨16067⟩⟩
def rawTerms : List Term := Proof.Events787.exact201538RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 201538
def producerEvent : Nat := 201537
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201538.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 201453, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult201538

namespace SemanticResult201543
def owner : Owner := ⟨.program ⟨257⟩, ⟨16068⟩⟩
def rawTerms : List Term := Proof.Events787.exact201543RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 201543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201543.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge201542.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge201542.frameStart)
    (transferEvent := 201541) (owner := owner)
    (leftResult := 201515) (rightResult := 201538)
    (working := LeftOperatorMerge201542.working)
    (reconstruction := LeftOperatorMerge201542.reconstruction)
    (leftReference := .predecessor 0 201539 .coefficient) (rightReference := .predecessor 1 201540 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult201515.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult201538.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge201542.operationAgreement
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
end SemanticResult201543

namespace SemanticResult201546
def owner : Owner := ⟨.program ⟨257⟩, ⟨7198⟩⟩
def rawTerms : List Term := Proof.Events787.exact201546RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 201546
def producerEvent : Nat := 201545
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201546.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 201453, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult201546

namespace SemanticResult201550
def owner : Owner := ⟨.program ⟨257⟩, ⟨16069⟩⟩
def rawTerms : List Term := Proof.Events787.exact201550RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 201550
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201550.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 201547) (rightBinding := 201548)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7198⟩) (rightExpression := ⟨16068⟩)
    (transferEvent := 201549)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201546.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult201543.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201550

namespace SemanticResult201554
def owner : Owner := ⟨.program ⟨257⟩, ⟨17821⟩⟩
def rawTerms : List Term := Proof.Events787.exact201554RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 201554
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201554.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 201551) (rightBinding := 201552)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16069⟩) (rightExpression := ⟨17818⟩)
    (transferEvent := 201553)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201550.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult201535.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201554

namespace SemanticResult201563
def owner : Owner := ⟨.program ⟨257⟩, ⟨16639⟩⟩
def rawTerms : List Term := Proof.Events787.exact201563RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 201563
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201563.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge201398.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge201398.frameStart)
    (owner := owner) (leftOwner := SemanticResult192995.owner)
    (rightOwner := SemanticResult201392.owner)
    (leftResult := 192995) (rightResult := 201392)
    (leftActual := SemanticResult192995.actual selector witness)
    (rightActual := SemanticResult201392.actual selector witness)
    (leftRaw := SemanticResult192995.rawTerms)
    (rightRaw := SemanticResult201392.rawTerms)
    (working := LeftOperatorMerge201398.working)
    (leftBinding := 201393) (rightBinding := 201394)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5909⟩) (rightExpression := ⟨16638⟩)
    (coefficientTransfer := 201395) (summaryTransfer := 201397)
    (rightCoefficientProducer := 201391)
    (rightSummaryTransfer := 201396)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge201398.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound201391.actual selector witness)
    (summaryMagnitude := LeftBound201397.actual selector witness)
    (reconstruction := LeftOperatorMerge201398.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult192995.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult201392.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound201391.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound201391.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge201398.operationAgreement
  · exact LeftBound201397.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge201398.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 201558 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17019⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨16067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17019⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge201398.working
    [{ coefficient := (1), key := LeftRelationMerge201558.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge201558.frameStart
      LeftRelationMerge201558.owner (.relation 201558) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge201558.deltas
    rows := LeftRelationMerge201558.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge201398.working LeftRelationMerge201558.source
        (relationContext LeftRelationMerge201558.source
          LeftRelationMerge201558.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge201398.working, LeftRelationMerge201558.deltas,
    LeftRelationMerge201558.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 201558)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16639⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16636⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16636⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge201398.working) (working := relationWorking0)
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
end SemanticResult201563

namespace SemanticResult201570
def owner : Owner := ⟨.program ⟨257⟩, ⟨17820⟩⟩
def rawTerms : List Term := Proof.Events787.exact201570RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 201570
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201570.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge201567.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult201563.owner)
    (rightOwner := SemanticResult201385.owner)
    (leftResult := 201563) (rightResult := 201385)
    (leftActual := SemanticResult201563.actual selector witness)
    (rightActual := SemanticResult201385.actual selector witness)
    (leftRaw := SemanticResult201563.rawTerms)
    (rightRaw := SemanticResult201385.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 201564) (rightBinding := 201565)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16639⟩) (rightExpression := ⟨17819⟩)
    (coefficientTransfer := 201566) (summaryTransfer := 201569)
    (base := LeftOperatorMerge201567.base)
    (reconstruction := LeftOperatorMerge201567.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201563.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult201385.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge201567.operationAgreement
  · rfl
  · decide
end SemanticResult201570

namespace SemanticResult201575
def owner : Owner := ⟨.program ⟨257⟩, ⟨20718⟩⟩
def rawTerms : List Term := Proof.Events787.exact201575RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 201575
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201575.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult201570.owner)
    (rightOwner := SemanticResult201088.owner)
    (leftResult := 201570) (rightResult := 201088)
    (leftActual := SemanticResult201570.actual selector witness)
    (rightActual := SemanticResult201088.actual selector witness)
    (leftRaw := SemanticResult201570.rawTerms)
    (rightRaw := SemanticResult201088.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 201571) (rightBinding := 201572)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17820⟩) (rightExpression := ⟨20717⟩)
    (transferEvent := 201573) (summaryTransferEvent := 201574)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201570.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult201088.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201575

namespace SemanticResult201580
def owner : Owner := ⟨.program ⟨257⟩, ⟨23938⟩⟩
def rawTerms : List Term := Proof.Events787.exact201580RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 201580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201580.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult201575.owner)
    (rightOwner := SemanticResult200606.owner)
    (leftResult := 201575) (rightResult := 200606)
    (leftActual := SemanticResult201575.actual selector witness)
    (rightActual := SemanticResult200606.actual selector witness)
    (leftRaw := SemanticResult201575.rawTerms)
    (rightRaw := SemanticResult200606.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 201576) (rightBinding := 201577)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20718⟩) (rightExpression := ⟨23937⟩)
    (transferEvent := 201578) (summaryTransferEvent := 201579)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201575.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult200606.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201580

namespace SemanticResult201585
def owner : Owner := ⟨.program ⟨257⟩, ⟨33958⟩⟩
def rawTerms : List Term := Proof.Events787.exact201585RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 201585
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201585.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult201580.owner)
    (rightOwner := SemanticResult200124.owner)
    (leftResult := 201580) (rightResult := 200124)
    (leftActual := SemanticResult201580.actual selector witness)
    (rightActual := SemanticResult200124.actual selector witness)
    (leftRaw := SemanticResult201580.rawTerms)
    (rightRaw := SemanticResult200124.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 201581) (rightBinding := 201582)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23938⟩) (rightExpression := ⟨33957⟩)
    (transferEvent := 201583) (summaryTransferEvent := 201584)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201580.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult200124.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201585

namespace SemanticResult201590
def owner : Owner := ⟨.program ⟨257⟩, ⟨53018⟩⟩
def rawTerms : List Term := Proof.Events787.exact201590RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 201590
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201590.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult201585.owner)
    (rightOwner := SemanticResult199642.owner)
    (leftResult := 201585) (rightResult := 199642)
    (leftActual := SemanticResult201585.actual selector witness)
    (rightActual := SemanticResult199642.actual selector witness)
    (leftRaw := SemanticResult201585.rawTerms)
    (rightRaw := SemanticResult199642.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 201586) (rightBinding := 201587)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33958⟩) (rightExpression := ⟨53017⟩)
    (transferEvent := 201588) (summaryTransferEvent := 201589)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201585.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult199642.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201590

namespace SemanticResult201595
def owner : Owner := ⟨.program ⟨257⟩, ⟨55998⟩⟩
def rawTerms : List Term := Proof.Events787.exact201595RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 201595
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201595.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult201590.owner)
    (rightOwner := SemanticResult199160.owner)
    (leftResult := 201590) (rightResult := 199160)
    (leftActual := SemanticResult201590.actual selector witness)
    (rightActual := SemanticResult199160.actual selector witness)
    (leftRaw := SemanticResult201590.rawTerms)
    (rightRaw := SemanticResult199160.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 201591) (rightBinding := 201592)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53018⟩) (rightExpression := ⟨55997⟩)
    (transferEvent := 201593) (summaryTransferEvent := 201594)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201590.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult199160.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201595

namespace SemanticResult201600
def owner : Owner := ⟨.program ⟨257⟩, ⟨58978⟩⟩
def rawTerms : List Term := Proof.Events787.exact201600RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 201600
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201600.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult201595.owner)
    (rightOwner := SemanticResult198678.owner)
    (leftResult := 201595) (rightResult := 198678)
    (leftActual := SemanticResult201595.actual selector witness)
    (rightActual := SemanticResult198678.actual selector witness)
    (leftRaw := SemanticResult201595.rawTerms)
    (rightRaw := SemanticResult198678.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 201596) (rightBinding := 201597)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55998⟩) (rightExpression := ⟨58977⟩)
    (transferEvent := 201598) (summaryTransferEvent := 201599)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201595.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult198678.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201600

namespace SemanticResult201605
def owner : Owner := ⟨.program ⟨257⟩, ⟨61958⟩⟩
def rawTerms : List Term := Proof.Events787.exact201605RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 201605
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201605.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult201600.owner)
    (rightOwner := SemanticResult198196.owner)
    (leftResult := 201600) (rightResult := 198196)
    (leftActual := SemanticResult201600.actual selector witness)
    (rightActual := SemanticResult198196.actual selector witness)
    (leftRaw := SemanticResult201600.rawTerms)
    (rightRaw := SemanticResult198196.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 201601) (rightBinding := 201602)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58978⟩) (rightExpression := ⟨61957⟩)
    (transferEvent := 201603) (summaryTransferEvent := 201604)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201600.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult198196.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201605

namespace SemanticResult201610
def owner : Owner := ⟨.program ⟨257⟩, ⟨64938⟩⟩
def rawTerms : List Term := Proof.Events787.exact201610RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 201610
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult201610.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult201605.owner)
    (rightOwner := SemanticResult197714.owner)
    (leftResult := 201605) (rightResult := 197714)
    (leftActual := SemanticResult201605.actual selector witness)
    (rightActual := SemanticResult197714.actual selector witness)
    (leftRaw := SemanticResult201605.rawTerms)
    (rightRaw := SemanticResult197714.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 201606) (rightBinding := 201607)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61958⟩) (rightExpression := ⟨64937⟩)
    (transferEvent := 201608) (summaryTransferEvent := 201609)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult201605.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult197714.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult201610

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
