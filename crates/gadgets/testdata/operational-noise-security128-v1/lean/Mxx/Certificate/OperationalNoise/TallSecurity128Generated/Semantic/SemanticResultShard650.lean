import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard650
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard127
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard552
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard553
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard643
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard644
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard646
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard647
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard648
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard649

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult89986
def owner : Owner := ⟨.program ⟨257⟩, ⟨16130⟩⟩
def rawTerms : List Term := Proof.Events351.exact89986RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89986
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult89986.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89983) (rightBinding := 89984)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7197⟩) (rightExpression := ⟨16129⟩)
    (transferEvent := 89985)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89982.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89979.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89986

namespace SemanticResult89990
def owner : Owner := ⟨.program ⟨257⟩, ⟨17928⟩⟩
def rawTerms : List Term := Proof.Events351.exact89990RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 89990
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult89990.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 89987) (rightBinding := 89988)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16130⟩) (rightExpression := ⟨17923⟩)
    (transferEvent := 89989)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89986.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89971.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult89990

namespace SemanticResult89999
def owner : Owner := ⟨.program ⟨257⟩, ⟨16715⟩⟩
def rawTerms : List Term := Proof.Events351.exact89999RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 89999
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult89999.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge89834.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge89834.frameStart)
    (owner := owner) (leftOwner := SemanticResult75995.owner)
    (rightOwner := SemanticResult89828.owner)
    (leftResult := 75995) (rightResult := 89828)
    (leftActual := SemanticResult75995.actual selector witness)
    (rightActual := SemanticResult89828.actual selector witness)
    (leftRaw := SemanticResult75995.rawTerms)
    (rightRaw := SemanticResult89828.rawTerms)
    (working := LeftOperatorMerge89834.working)
    (leftBinding := 89829) (rightBinding := 89830)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10368⟩) (rightExpression := ⟨16714⟩)
    (coefficientTransfer := 89831) (summaryTransfer := 89833)
    (rightCoefficientProducer := 89827)
    (rightSummaryTransfer := 89832)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge89834.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound89827.actual selector witness)
    (summaryMagnitude := LeftBound89833.actual selector witness)
    (reconstruction := LeftOperatorMerge89834.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult75995.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89828.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89827.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound89827.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge89834.operationAgreement
  · exact LeftBound89833.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge89834.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 89994 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨15836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17054⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17922⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17054⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge89834.working
    [{ coefficient := (1), key := LeftRelationMerge89994.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge89994.frameStart
      LeftRelationMerge89994.owner (.relation 89994) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge89994.deltas
    rows := LeftRelationMerge89994.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge89834.working LeftRelationMerge89994.source
        (relationContext LeftRelationMerge89994.source
          LeftRelationMerge89994.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge89834.working, LeftRelationMerge89994.deltas,
    LeftRelationMerge89994.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 89994)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16715⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16712⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16712⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge89834.working) (working := relationWorking0)
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
end SemanticResult89999

namespace SemanticResult90006
def owner : Owner := ⟨.program ⟨257⟩, ⟨17925⟩⟩
def rawTerms : List Term := Proof.Events351.exact90006RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 90006
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90006.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge90003.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult89999.owner)
    (rightOwner := SemanticResult89821.owner)
    (leftResult := 89999) (rightResult := 89821)
    (leftActual := SemanticResult89999.actual selector witness)
    (rightActual := SemanticResult89821.actual selector witness)
    (leftRaw := SemanticResult89999.rawTerms)
    (rightRaw := SemanticResult89821.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90000) (rightBinding := 90001)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16715⟩) (rightExpression := ⟨17924⟩)
    (coefficientTransfer := 90002) (summaryTransfer := 90005)
    (base := LeftOperatorMerge90003.base)
    (reconstruction := LeftOperatorMerge90003.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult89999.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89821.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge90003.operationAgreement
  · rfl
  · decide
end SemanticResult90006

namespace SemanticResult90016
def owner : Owner := ⟨.program ⟨257⟩, ⟨17926⟩⟩
def rawTerms : List Term := Proof.Events351.exact90016RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529920)
def resultEvent : Nat := 90016
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90016.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨32188807212483706889510625476608, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge90012.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge90012.frameStart)
    (owner := owner) (leftOwner := SemanticResult90006.owner)
    (rightOwner := SemanticResult15882.owner)
    (leftResult := 90006) (rightResult := 15882)
    (leftActual := SemanticResult90006.actual selector witness)
    (rightActual := SemanticResult15882.actual selector witness)
    (leftRaw := SemanticResult90006.rawTerms)
    (rightRaw := SemanticResult15882.rawTerms)
    (working := LeftOperatorMerge90012.working)
    (leftBinding := 90007) (rightBinding := 90008)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17925⟩) (rightExpression := ⟨7172⟩)
    (coefficientTransfer := 90009) (summaryTransfer := 90011)
    (rightCoefficientProducer := 15881)
    (rightSummaryTransfer := 90010)
    (leftMaximum := ⟨32188807212483706889510625476608, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge90012.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15881.actual selector witness)
    (summaryMagnitude := LeftBound90011.actual selector witness)
    (reconstruction := LeftOperatorMerge90012.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90006.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15882.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15881.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15881.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge90012.operationAgreement
  · exact LeftBound90011.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge90012.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 90014 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge90012.working
    [{ coefficient := (-1), key := LeftRelationMerge90014.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge90014.frameStart
      LeftRelationMerge90014.owner (.relation 90014) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge90014.deltas
    rows := LeftRelationMerge90014.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge90012.working LeftRelationMerge90014.source
        (relationContext LeftRelationMerge90014.source
          LeftRelationMerge90014.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge90012.working, LeftRelationMerge90014.deltas,
    LeftRelationMerge90014.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 90014)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨17926⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge90012.working) (working := relationWorking0)
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
end SemanticResult90016

namespace SemanticResult90021
def owner : Owner := ⟨.program ⟨257⟩, ⟨10372⟩⟩
def rawTerms : List Term := Proof.Events351.exact90021RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 90021
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90021.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge90020.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge90020.frameStart)
    (transferEvent := 90019) (owner := owner)
    (leftResult := 723) (rightResult := 75903)
    (working := LeftOperatorMerge90020.working)
    (reconstruction := LeftOperatorMerge90020.reconstruction)
    (leftReference := .predecessor 0 90017 .coefficient) (rightReference := .predecessor 1 90018 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult75903.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge90020.operationAgreement
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
end SemanticResult90021

namespace SemanticResult90026
def owner : Owner := ⟨.program ⟨257⟩, ⟨10350⟩⟩
def rawTerms : List Term := Proof.Events351.exact90026RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 90026
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90026.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge90025.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge90025.frameStart)
    (transferEvent := 90024) (owner := owner)
    (leftResult := 75773) (rightResult := 15896)
    (working := LeftOperatorMerge90025.working)
    (reconstruction := LeftOperatorMerge90025.reconstruction)
    (leftReference := .predecessor 0 90022 .coefficient) (rightReference := .predecessor 1 90023 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult75773.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15896.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge90025.operationAgreement
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
end SemanticResult90026

namespace SemanticResult90030
def owner : Owner := ⟨.program ⟨257⟩, ⟨10373⟩⟩
def rawTerms : List Term := Proof.Events351.exact90030RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 90030
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90030.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 90027) (rightBinding := 90028)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10350⟩) (rightExpression := ⟨10372⟩)
    (transferEvent := 90029)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90026.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult90021.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90030

namespace SemanticResult90036
def owner : Owner := ⟨.program ⟨257⟩, ⟨10374⟩⟩
def rawTerms : List Term := Proof.Events351.exact90036RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 90036
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90036.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 90033) (survivorTransfer := 90034)
    (survivorEvent := 90035) (resultEvent := resultEvent)
    (rightCoefficientProducer := 31515)
    (owner := owner) (leftOwner := SemanticResult90030.owner)
    (rightOwner := SemanticResult31516.owner)
    (leftResult := 90030) (rightResult := 31516)
    (leftBinding := 90031) (rightBinding := 90032)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10373⟩) (rightExpression := ⟨118⟩)
    (leftActual := SemanticResult90030.actual selector witness)
    (rightActual := SemanticResult31516.actual selector witness)
    (leftRaw := SemanticResult90030.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound31515.actual selector witness)
    (survivorMagnitude := LeftBound90034.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90030.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)
  · exact LeftBound90034.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult90036

namespace SemanticResult90043
def owner : Owner := ⟨.program ⟨257⟩, ⟨10375⟩⟩
def rawTerms : List Term := Proof.Events351.exact90043RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 90043
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90043.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge90040.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90036.owner)
    (rightOwner := SemanticResult90036.owner)
    (leftResult := 90036) (rightResult := 90036)
    (leftActual := SemanticResult90036.actual selector witness)
    (rightActual := SemanticResult90036.actual selector witness)
    (leftRaw := SemanticResult90036.rawTerms)
    (rightRaw := SemanticResult90036.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90037) (rightBinding := 90038)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10374⟩) (rightExpression := ⟨10374⟩)
    (coefficientTransfer := 90039) (summaryTransfer := 90042)
    (base := LeftOperatorMerge90040.base)
    (reconstruction := LeftOperatorMerge90040.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90036.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult90036.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge90040.operationAgreement
  · rfl
  · decide
end SemanticResult90043

namespace SemanticResult90048
def owner : Owner := ⟨.program ⟨257⟩, ⟨17927⟩⟩
def rawTerms : List Term := Proof.Events351.exact90048RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 90048
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90048.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90043.owner)
    (rightOwner := SemanticResult90016.owner)
    (leftResult := 90043) (rightResult := 90016)
    (leftActual := SemanticResult90043.actual selector witness)
    (rightActual := SemanticResult90016.actual selector witness)
    (leftRaw := SemanticResult90043.rawTerms)
    (rightRaw := SemanticResult90016.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90044) (rightBinding := 90045)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨10375⟩) (rightExpression := ⟨17926⟩)
    (transferEvent := 90046) (summaryTransferEvent := 90047)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90043.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult90016.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90048

namespace SemanticResult90053
def owner : Owner := ⟨.program ⟨257⟩, ⟨20836⟩⟩
def rawTerms : List Term := Proof.Events351.exact90053RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 90053
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90053.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90048.owner)
    (rightOwner := SemanticResult89804.owner)
    (leftResult := 90048) (rightResult := 89804)
    (leftActual := SemanticResult90048.actual selector witness)
    (rightActual := SemanticResult89804.actual selector witness)
    (leftRaw := SemanticResult90048.rawTerms)
    (rightRaw := SemanticResult89804.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90049) (rightBinding := 90050)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17927⟩) (rightExpression := ⟨20835⟩)
    (transferEvent := 90051) (summaryTransferEvent := 90052)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90048.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89804.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90053

namespace SemanticResult90058
def owner : Owner := ⟨.program ⟨257⟩, ⟨24056⟩⟩
def rawTerms : List Term := Proof.Events351.exact90058RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 90058
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90058.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90053.owner)
    (rightOwner := SemanticResult89592.owner)
    (leftResult := 90053) (rightResult := 89592)
    (leftActual := SemanticResult90053.actual selector witness)
    (rightActual := SemanticResult89592.actual selector witness)
    (leftRaw := SemanticResult90053.rawTerms)
    (rightRaw := SemanticResult89592.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90054) (rightBinding := 90055)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20836⟩) (rightExpression := ⟨24055⟩)
    (transferEvent := 90056) (summaryTransferEvent := 90057)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90053.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89592.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90058

namespace SemanticResult90063
def owner : Owner := ⟨.program ⟨257⟩, ⟨34076⟩⟩
def rawTerms : List Term := Proof.Events351.exact90063RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 90063
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90063.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90058.owner)
    (rightOwner := SemanticResult89380.owner)
    (leftResult := 90058) (rightResult := 89380)
    (leftActual := SemanticResult90058.actual selector witness)
    (rightActual := SemanticResult89380.actual selector witness)
    (leftRaw := SemanticResult90058.rawTerms)
    (rightRaw := SemanticResult89380.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90059) (rightBinding := 90060)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24056⟩) (rightExpression := ⟨34075⟩)
    (transferEvent := 90061) (summaryTransferEvent := 90062)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90058.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89380.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90063

namespace SemanticResult90068
def owner : Owner := ⟨.program ⟨257⟩, ⟨53136⟩⟩
def rawTerms : List Term := Proof.Events351.exact90068RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 90068
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90068.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90063.owner)
    (rightOwner := SemanticResult89168.owner)
    (leftResult := 90063) (rightResult := 89168)
    (leftActual := SemanticResult90063.actual selector witness)
    (rightActual := SemanticResult89168.actual selector witness)
    (leftRaw := SemanticResult90063.rawTerms)
    (rightRaw := SemanticResult89168.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90064) (rightBinding := 90065)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34076⟩) (rightExpression := ⟨53135⟩)
    (transferEvent := 90066) (summaryTransferEvent := 90067)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90063.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult89168.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90068

namespace SemanticResult90073
def owner : Owner := ⟨.program ⟨257⟩, ⟨56116⟩⟩
def rawTerms : List Term := Proof.Events351.exact90073RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 90073
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult90073.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult90068.owner)
    (rightOwner := SemanticResult88956.owner)
    (leftResult := 90068) (rightResult := 88956)
    (leftActual := SemanticResult90068.actual selector witness)
    (rightActual := SemanticResult88956.actual selector witness)
    (leftRaw := SemanticResult90068.rawTerms)
    (rightRaw := SemanticResult88956.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 90069) (rightBinding := 90070)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53136⟩) (rightExpression := ⟨56115⟩)
    (transferEvent := 90071) (summaryTransferEvent := 90072)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90068.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult88956.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult90073

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
