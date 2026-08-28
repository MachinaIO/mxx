import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1555
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard127
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1457
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1458
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1549
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1551
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1552
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1553
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1554

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult221607
def owner : Owner := ⟨.program ⟨257⟩, ⟨7197⟩⟩
def rawTerms : List Term := Proof.Events865.exact221607RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 221607
def producerEvent : Nat := 221606
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221607.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 221514, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult221607

namespace SemanticResult221611
def owner : Owner := ⟨.program ⟨257⟩, ⟨16034⟩⟩
def rawTerms : List Term := Proof.Events865.exact221611RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 221611
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221611.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 221608) (rightBinding := 221609)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7197⟩) (rightExpression := ⟨16033⟩)
    (transferEvent := 221610)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221607.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221604.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221611

namespace SemanticResult221615
def owner : Owner := ⟨.program ⟨257⟩, ⟨17760⟩⟩
def rawTerms : List Term := Proof.Events865.exact221615RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 221615
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221615.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 221612) (rightBinding := 221613)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16034⟩) (rightExpression := ⟨17755⟩)
    (transferEvent := 221614)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221611.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221596.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221615

namespace SemanticResult221624
def owner : Owner := ⟨.program ⟨257⟩, ⟨16595⟩⟩
def rawTerms : List Term := Proof.Events865.exact221624RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 221624
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221624.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge221459.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge221459.frameStart)
    (owner := owner) (leftOwner := SemanticResult207620.owner)
    (rightOwner := SemanticResult221453.owner)
    (leftResult := 207620) (rightResult := 221453)
    (leftActual := SemanticResult207620.actual selector witness)
    (rightActual := SemanticResult221453.actual selector witness)
    (leftRaw := SemanticResult207620.rawTerms)
    (rightRaw := SemanticResult221453.rawTerms)
    (working := LeftOperatorMerge221459.working)
    (leftBinding := 221454) (rightBinding := 221455)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5599⟩) (rightExpression := ⟨16594⟩)
    (coefficientTransfer := 221456) (summaryTransfer := 221458)
    (rightCoefficientProducer := 221452)
    (rightSummaryTransfer := 221457)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge221459.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound221452.actual selector witness)
    (summaryMagnitude := LeftBound221458.actual selector witness)
    (reconstruction := LeftOperatorMerge221459.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207620.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221453.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound221452.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound221452.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge221459.operationAgreement
  · exact LeftBound221458.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge221459.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 221619 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17754⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17000⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17754⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17000⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge221459.working
    [{ coefficient := (1), key := LeftRelationMerge221619.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge221619.frameStart
      LeftRelationMerge221619.owner (.relation 221619) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge221619.deltas
    rows := LeftRelationMerge221619.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge221459.working LeftRelationMerge221619.source
        (relationContext LeftRelationMerge221619.source
          LeftRelationMerge221619.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge221459.working, LeftRelationMerge221619.deltas,
    LeftRelationMerge221619.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 221619)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16595⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16592⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16592⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge221459.working) (working := relationWorking0)
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
end SemanticResult221624

namespace SemanticResult221631
def owner : Owner := ⟨.program ⟨257⟩, ⟨17757⟩⟩
def rawTerms : List Term := Proof.Events865.exact221631RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 221631
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221631.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge221628.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221624.owner)
    (rightOwner := SemanticResult221446.owner)
    (leftResult := 221624) (rightResult := 221446)
    (leftActual := SemanticResult221624.actual selector witness)
    (rightActual := SemanticResult221446.actual selector witness)
    (leftRaw := SemanticResult221624.rawTerms)
    (rightRaw := SemanticResult221446.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221625) (rightBinding := 221626)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16595⟩) (rightExpression := ⟨17756⟩)
    (coefficientTransfer := 221627) (summaryTransfer := 221630)
    (base := LeftOperatorMerge221628.base)
    (reconstruction := LeftOperatorMerge221628.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221624.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221446.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge221628.operationAgreement
  · rfl
  · decide
end SemanticResult221631

namespace SemanticResult221641
def owner : Owner := ⟨.program ⟨257⟩, ⟨17758⟩⟩
def rawTerms : List Term := Proof.Events865.exact221641RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529920)
def resultEvent : Nat := 221641
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221641.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨32188807212483706889510625476608, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge221637.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge221637.frameStart)
    (owner := owner) (leftOwner := SemanticResult221631.owner)
    (rightOwner := SemanticResult15882.owner)
    (leftResult := 221631) (rightResult := 15882)
    (leftActual := SemanticResult221631.actual selector witness)
    (rightActual := SemanticResult15882.actual selector witness)
    (leftRaw := SemanticResult221631.rawTerms)
    (rightRaw := SemanticResult15882.rawTerms)
    (working := LeftOperatorMerge221637.working)
    (leftBinding := 221632) (rightBinding := 221633)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17757⟩) (rightExpression := ⟨7172⟩)
    (coefficientTransfer := 221634) (summaryTransfer := 221636)
    (rightCoefficientProducer := 15881)
    (rightSummaryTransfer := 221635)
    (leftMaximum := ⟨32188807212483706889510625476608, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge221637.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15881.actual selector witness)
    (summaryMagnitude := LeftBound221636.actual selector witness)
    (reconstruction := LeftOperatorMerge221637.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221631.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15882.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15881.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15881.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge221637.operationAgreement
  · exact LeftBound221636.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge221637.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 221639 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge221637.working
    [{ coefficient := (-1), key := LeftRelationMerge221639.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge221639.frameStart
      LeftRelationMerge221639.owner (.relation 221639) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge221639.deltas
    rows := LeftRelationMerge221639.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge221637.working LeftRelationMerge221639.source
        (relationContext LeftRelationMerge221639.source
          LeftRelationMerge221639.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge221637.working, LeftRelationMerge221639.deltas,
    LeftRelationMerge221639.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 221639)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨17758⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge221637.working) (working := relationWorking0)
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
end SemanticResult221641

namespace SemanticResult221646
def owner : Owner := ⟨.program ⟨257⟩, ⟨7085⟩⟩
def rawTerms : List Term := Proof.Events865.exact221646RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 221646
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221646.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge221645.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge221645.frameStart)
    (transferEvent := 221644) (owner := owner)
    (leftResult := 723) (rightResult := 207528)
    (working := LeftOperatorMerge221645.working)
    (reconstruction := LeftOperatorMerge221645.reconstruction)
    (leftReference := .predecessor 0 221642 .coefficient) (rightReference := .predecessor 1 221643 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult207528.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge221645.operationAgreement
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
end SemanticResult221646

namespace SemanticResult221651
def owner : Owner := ⟨.program ⟨257⟩, ⟨8598⟩⟩
def rawTerms : List Term := Proof.Events865.exact221651RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 221651
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221651.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge221650.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge221650.frameStart)
    (transferEvent := 221649) (owner := owner)
    (leftResult := 207398) (rightResult := 15896)
    (working := LeftOperatorMerge221650.working)
    (reconstruction := LeftOperatorMerge221650.reconstruction)
    (leftReference := .predecessor 0 221647 .coefficient) (rightReference := .predecessor 1 221648 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult207398.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15896.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge221650.operationAgreement
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
end SemanticResult221651

namespace SemanticResult221655
def owner : Owner := ⟨.program ⟨257⟩, ⟨9389⟩⟩
def rawTerms : List Term := Proof.Events865.exact221655RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 221655
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221655.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 221652) (rightBinding := 221653)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8598⟩) (rightExpression := ⟨7085⟩)
    (transferEvent := 221654)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221651.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221646.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221655

namespace SemanticResult221661
def owner : Owner := ⟨.program ⟨257⟩, ⟨9390⟩⟩
def rawTerms : List Term := Proof.Events865.exact221661RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 221661
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221661.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 221658) (survivorTransfer := 221659)
    (survivorEvent := 221660) (resultEvent := resultEvent)
    (rightCoefficientProducer := 31515)
    (owner := owner) (leftOwner := SemanticResult221655.owner)
    (rightOwner := SemanticResult31516.owner)
    (leftResult := 221655) (rightResult := 31516)
    (leftBinding := 221656) (rightBinding := 221657)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9389⟩) (rightExpression := ⟨118⟩)
    (leftActual := SemanticResult221655.actual selector witness)
    (rightActual := SemanticResult31516.actual selector witness)
    (leftRaw := SemanticResult221655.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound31515.actual selector witness)
    (survivorMagnitude := LeftBound221659.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221655.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)
  · exact LeftBound221659.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult221661

namespace SemanticResult221668
def owner : Owner := ⟨.program ⟨257⟩, ⟨9477⟩⟩
def rawTerms : List Term := Proof.Events865.exact221668RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 221668
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221668.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge221665.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221661.owner)
    (rightOwner := SemanticResult221661.owner)
    (leftResult := 221661) (rightResult := 221661)
    (leftActual := SemanticResult221661.actual selector witness)
    (rightActual := SemanticResult221661.actual selector witness)
    (leftRaw := SemanticResult221661.rawTerms)
    (rightRaw := SemanticResult221661.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221662) (rightBinding := 221663)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9390⟩) (rightExpression := ⟨9390⟩)
    (coefficientTransfer := 221664) (summaryTransfer := 221667)
    (base := LeftOperatorMerge221665.base)
    (reconstruction := LeftOperatorMerge221665.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221661.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221661.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge221665.operationAgreement
  · rfl
  · decide
end SemanticResult221668

namespace SemanticResult221673
def owner : Owner := ⟨.program ⟨257⟩, ⟨17759⟩⟩
def rawTerms : List Term := Proof.Events865.exact221673RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 221673
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221673.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221668.owner)
    (rightOwner := SemanticResult221641.owner)
    (leftResult := 221668) (rightResult := 221641)
    (leftActual := SemanticResult221668.actual selector witness)
    (rightActual := SemanticResult221641.actual selector witness)
    (leftRaw := SemanticResult221668.rawTerms)
    (rightRaw := SemanticResult221641.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221669) (rightBinding := 221670)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9477⟩) (rightExpression := ⟨17758⟩)
    (transferEvent := 221671) (summaryTransferEvent := 221672)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221668.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221641.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221673

namespace SemanticResult221678
def owner : Owner := ⟨.program ⟨257⟩, ⟨20650⟩⟩
def rawTerms : List Term := Proof.Events865.exact221678RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 221678
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221678.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221673.owner)
    (rightOwner := SemanticResult221429.owner)
    (leftResult := 221673) (rightResult := 221429)
    (leftActual := SemanticResult221673.actual selector witness)
    (rightActual := SemanticResult221429.actual selector witness)
    (leftRaw := SemanticResult221673.rawTerms)
    (rightRaw := SemanticResult221429.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221674) (rightBinding := 221675)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17759⟩) (rightExpression := ⟨20649⟩)
    (transferEvent := 221676) (summaryTransferEvent := 221677)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221673.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221429.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221678

namespace SemanticResult221683
def owner : Owner := ⟨.program ⟨257⟩, ⟨23870⟩⟩
def rawTerms : List Term := Proof.Events865.exact221683RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 221683
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221683.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221678.owner)
    (rightOwner := SemanticResult221217.owner)
    (leftResult := 221678) (rightResult := 221217)
    (leftActual := SemanticResult221678.actual selector witness)
    (rightActual := SemanticResult221217.actual selector witness)
    (leftRaw := SemanticResult221678.rawTerms)
    (rightRaw := SemanticResult221217.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221679) (rightBinding := 221680)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20650⟩) (rightExpression := ⟨23869⟩)
    (transferEvent := 221681) (summaryTransferEvent := 221682)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221678.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221217.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221683

namespace SemanticResult221688
def owner : Owner := ⟨.program ⟨257⟩, ⟨33890⟩⟩
def rawTerms : List Term := Proof.Events865.exact221688RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 221688
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221688.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221683.owner)
    (rightOwner := SemanticResult221005.owner)
    (leftResult := 221683) (rightResult := 221005)
    (leftActual := SemanticResult221683.actual selector witness)
    (rightActual := SemanticResult221005.actual selector witness)
    (leftRaw := SemanticResult221683.rawTerms)
    (rightRaw := SemanticResult221005.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221684) (rightBinding := 221685)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23870⟩) (rightExpression := ⟨33889⟩)
    (transferEvent := 221686) (summaryTransferEvent := 221687)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221683.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult221005.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221688

namespace SemanticResult221693
def owner : Owner := ⟨.program ⟨257⟩, ⟨52950⟩⟩
def rawTerms : List Term := Proof.Events865.exact221693RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 221693
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult221693.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult221688.owner)
    (rightOwner := SemanticResult220793.owner)
    (leftResult := 221688) (rightResult := 220793)
    (leftActual := SemanticResult221688.actual selector witness)
    (rightActual := SemanticResult220793.actual selector witness)
    (leftRaw := SemanticResult221688.rawTerms)
    (rightRaw := SemanticResult220793.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 221689) (rightBinding := 221690)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33890⟩) (rightExpression := ⟨52949⟩)
    (transferEvent := 221691) (summaryTransferEvent := 221692)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult221688.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult220793.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult221693

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
