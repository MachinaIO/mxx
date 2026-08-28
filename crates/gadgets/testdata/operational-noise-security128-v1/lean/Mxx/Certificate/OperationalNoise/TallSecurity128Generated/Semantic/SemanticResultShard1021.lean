import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1021
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard955
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard976
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard980
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard984
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard988
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard991
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard995
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard999
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1002
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1010
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1013
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1017
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1020

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult143050
def owner : Owner := ⟨.program ⟨257⟩, ⟨15925⟩⟩
def rawTerms : List Term := Proof.Events558.exact143050RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 143050
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143050.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 143047) (rightBinding := 143048)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7198⟩) (rightExpression := ⟨15924⟩)
    (transferEvent := 143049)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143046.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult143043.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143050

namespace SemanticResult143054
def owner : Owner := ⟨.program ⟨257⟩, ⟨17569⟩⟩
def rawTerms : List Term := Proof.Events558.exact143054RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 143054
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143054.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 143051) (rightBinding := 143052)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15925⟩) (rightExpression := ⟨17566⟩)
    (transferEvent := 143053)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143050.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult143035.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143054

namespace SemanticResult143063
def owner : Owner := ⟨.program ⟨257⟩, ⟨16459⟩⟩
def rawTerms : List Term := Proof.Events558.exact143063RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 143063
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143063.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge142898.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge142898.frameStart)
    (owner := owner) (leftOwner := SemanticResult134495.owner)
    (rightOwner := SemanticResult142892.owner)
    (leftResult := 134495) (rightResult := 142892)
    (leftActual := SemanticResult134495.actual selector witness)
    (rightActual := SemanticResult142892.actual selector witness)
    (leftRaw := SemanticResult134495.rawTerms)
    (rightRaw := SemanticResult142892.rawTerms)
    (working := LeftOperatorMerge142898.working)
    (leftBinding := 142893) (rightBinding := 142894)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5473⟩) (rightExpression := ⟨16458⟩)
    (coefficientTransfer := 142895) (summaryTransfer := 142897)
    (rightCoefficientProducer := 142891)
    (rightSummaryTransfer := 142896)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge142898.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound142891.actual selector witness)
    (summaryMagnitude := LeftBound142897.actual selector witness)
    (reconstruction := LeftOperatorMerge142898.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult134495.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult142892.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142891.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound142891.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge142898.operationAgreement
  · exact LeftBound142897.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge142898.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 143058 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16938⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨15923⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17565⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16938⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15923⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge142898.working
    [{ coefficient := (1), key := LeftRelationMerge143058.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge143058.frameStart
      LeftRelationMerge143058.owner (.relation 143058) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge143058.deltas
    rows := LeftRelationMerge143058.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge142898.working LeftRelationMerge143058.source
        (relationContext LeftRelationMerge143058.source
          LeftRelationMerge143058.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge142898.working, LeftRelationMerge143058.deltas,
    LeftRelationMerge143058.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 143058)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16459⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16456⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge142898.working) (working := relationWorking0)
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
end SemanticResult143063

namespace SemanticResult143070
def owner : Owner := ⟨.program ⟨257⟩, ⟨17568⟩⟩
def rawTerms : List Term := Proof.Events558.exact143070RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 143070
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143070.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge143067.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143063.owner)
    (rightOwner := SemanticResult142885.owner)
    (leftResult := 143063) (rightResult := 142885)
    (leftActual := SemanticResult143063.actual selector witness)
    (rightActual := SemanticResult142885.actual selector witness)
    (leftRaw := SemanticResult143063.rawTerms)
    (rightRaw := SemanticResult142885.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143064) (rightBinding := 143065)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16459⟩) (rightExpression := ⟨17567⟩)
    (coefficientTransfer := 143066) (summaryTransfer := 143069)
    (base := LeftOperatorMerge143067.base)
    (reconstruction := LeftOperatorMerge143067.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143063.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult142885.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge143067.operationAgreement
  · rfl
  · decide
end SemanticResult143070

namespace SemanticResult143075
def owner : Owner := ⟨.program ⟨257⟩, ⟨20439⟩⟩
def rawTerms : List Term := Proof.Events558.exact143075RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 143075
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143075.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143070.owner)
    (rightOwner := SemanticResult142588.owner)
    (leftResult := 143070) (rightResult := 142588)
    (leftActual := SemanticResult143070.actual selector witness)
    (rightActual := SemanticResult142588.actual selector witness)
    (leftRaw := SemanticResult143070.rawTerms)
    (rightRaw := SemanticResult142588.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143071) (rightBinding := 143072)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17568⟩) (rightExpression := ⟨20438⟩)
    (transferEvent := 143073) (summaryTransferEvent := 143074)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143070.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult142588.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143075

namespace SemanticResult143080
def owner : Owner := ⟨.program ⟨257⟩, ⟨23659⟩⟩
def rawTerms : List Term := Proof.Events558.exact143080RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 143080
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143080.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143075.owner)
    (rightOwner := SemanticResult142106.owner)
    (leftResult := 143075) (rightResult := 142106)
    (leftActual := SemanticResult143075.actual selector witness)
    (rightActual := SemanticResult142106.actual selector witness)
    (leftRaw := SemanticResult143075.rawTerms)
    (rightRaw := SemanticResult142106.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143076) (rightBinding := 143077)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20439⟩) (rightExpression := ⟨23658⟩)
    (transferEvent := 143078) (summaryTransferEvent := 143079)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143075.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult142106.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143080

namespace SemanticResult143085
def owner : Owner := ⟨.program ⟨257⟩, ⟨33679⟩⟩
def rawTerms : List Term := Proof.Events558.exact143085RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 143085
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143085.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143080.owner)
    (rightOwner := SemanticResult141624.owner)
    (leftResult := 143080) (rightResult := 141624)
    (leftActual := SemanticResult143080.actual selector witness)
    (rightActual := SemanticResult141624.actual selector witness)
    (leftRaw := SemanticResult143080.rawTerms)
    (rightRaw := SemanticResult141624.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143081) (rightBinding := 143082)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23659⟩) (rightExpression := ⟨33678⟩)
    (transferEvent := 143083) (summaryTransferEvent := 143084)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143080.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult141624.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143085

namespace SemanticResult143090
def owner : Owner := ⟨.program ⟨257⟩, ⟨52739⟩⟩
def rawTerms : List Term := Proof.Events558.exact143090RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 143090
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143090.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143085.owner)
    (rightOwner := SemanticResult141142.owner)
    (leftResult := 143085) (rightResult := 141142)
    (leftActual := SemanticResult143085.actual selector witness)
    (rightActual := SemanticResult141142.actual selector witness)
    (leftRaw := SemanticResult143085.rawTerms)
    (rightRaw := SemanticResult141142.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143086) (rightBinding := 143087)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33679⟩) (rightExpression := ⟨52738⟩)
    (transferEvent := 143088) (summaryTransferEvent := 143089)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143085.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult141142.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143090

namespace SemanticResult143095
def owner : Owner := ⟨.program ⟨257⟩, ⟨55719⟩⟩
def rawTerms : List Term := Proof.Events558.exact143095RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 143095
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143095.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143090.owner)
    (rightOwner := SemanticResult140660.owner)
    (leftResult := 143090) (rightResult := 140660)
    (leftActual := SemanticResult143090.actual selector witness)
    (rightActual := SemanticResult140660.actual selector witness)
    (leftRaw := SemanticResult143090.rawTerms)
    (rightRaw := SemanticResult140660.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143091) (rightBinding := 143092)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52739⟩) (rightExpression := ⟨55718⟩)
    (transferEvent := 143093) (summaryTransferEvent := 143094)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143090.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult140660.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143095

namespace SemanticResult143100
def owner : Owner := ⟨.program ⟨257⟩, ⟨58699⟩⟩
def rawTerms : List Term := Proof.Events558.exact143100RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 143100
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143100.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143095.owner)
    (rightOwner := SemanticResult140178.owner)
    (leftResult := 143095) (rightResult := 140178)
    (leftActual := SemanticResult143095.actual selector witness)
    (rightActual := SemanticResult140178.actual selector witness)
    (leftRaw := SemanticResult143095.rawTerms)
    (rightRaw := SemanticResult140178.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143096) (rightBinding := 143097)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55719⟩) (rightExpression := ⟨58698⟩)
    (transferEvent := 143098) (summaryTransferEvent := 143099)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143095.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult140178.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143100

namespace SemanticResult143105
def owner : Owner := ⟨.program ⟨257⟩, ⟨61679⟩⟩
def rawTerms : List Term := Proof.Events559.exact143105RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 143105
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143105.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143100.owner)
    (rightOwner := SemanticResult139696.owner)
    (leftResult := 143100) (rightResult := 139696)
    (leftActual := SemanticResult143100.actual selector witness)
    (rightActual := SemanticResult139696.actual selector witness)
    (leftRaw := SemanticResult143100.rawTerms)
    (rightRaw := SemanticResult139696.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143101) (rightBinding := 143102)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58699⟩) (rightExpression := ⟨61678⟩)
    (transferEvent := 143103) (summaryTransferEvent := 143104)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143100.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult139696.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143105

namespace SemanticResult143110
def owner : Owner := ⟨.program ⟨257⟩, ⟨64659⟩⟩
def rawTerms : List Term := Proof.Events559.exact143110RawTerms
def summary : Bound := (.finite 289706631804066638652128995049472)
def resultEvent : Nat := 143110
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143110.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143105.owner)
    (rightOwner := SemanticResult139214.owner)
    (leftResult := 143105) (rightResult := 139214)
    (leftActual := SemanticResult143105.actual selector witness)
    (rightActual := SemanticResult139214.actual selector witness)
    (leftRaw := SemanticResult143105.rawTerms)
    (rightRaw := SemanticResult139214.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 257515860087126057990209472036864)
    (rightMaximum := 32190771716940580661919523012608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143106) (rightBinding := 143107)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61679⟩) (rightExpression := ⟨64658⟩)
    (transferEvent := 143108) (summaryTransferEvent := 143109)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143105.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult139214.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143110

namespace SemanticResult143115
def owner : Owner := ⟨.program ⟨257⟩, ⟨69628⟩⟩
def rawTerms : List Term := Proof.Events559.exact143115RawTerms
def summary : Bound := (.finite 321897992872344281445771187322880)
def resultEvent : Nat := 143115
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143115.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143110.owner)
    (rightOwner := SemanticResult138732.owner)
    (leftResult := 143110) (rightResult := 138732)
    (leftActual := SemanticResult143110.actual selector witness)
    (rightActual := SemanticResult138732.actual selector witness)
    (leftRaw := SemanticResult143110.rawTerms)
    (rightRaw := SemanticResult138732.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 289706631804066638652128995049472)
    (rightMaximum := 32191361068277642793642192273408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143111) (rightBinding := 143112)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨64659⟩) (rightExpression := ⟨69627⟩)
    (transferEvent := 143113) (summaryTransferEvent := 143114)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143110.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult138732.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143115

namespace SemanticResult143120
def owner : Owner := ⟨.program ⟨257⟩, ⟨69629⟩⟩
def rawTerms : List Term := Proof.Events559.exact143120RawTerms
def summary : Bound := (.finite 354089550391067611616654269349888)
def resultEvent : Nat := 143120
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143120.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143115.owner)
    (rightOwner := SemanticResult138250.owner)
    (leftResult := 143115) (rightResult := 138250)
    (leftActual := SemanticResult143115.actual selector witness)
    (rightActual := SemanticResult138250.actual selector witness)
    (leftRaw := SemanticResult143115.rawTerms)
    (rightRaw := SemanticResult138250.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 321897992872344281445771187322880)
    (rightMaximum := 32191557518723330170883082027008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143116) (rightBinding := 143117)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69628⟩) (rightExpression := ⟨28117⟩)
    (transferEvent := 143118) (summaryTransferEvent := 143119)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143115.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult138250.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143120

namespace SemanticResult143125
def owner : Owner := ⟨.program ⟨257⟩, ⟨69630⟩⟩
def rawTerms : List Term := Proof.Events559.exact143125RawTerms
def summary : Bound := (.finite 386281697261128003919260020637696)
def resultEvent : Nat := 143125
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143125.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143120.owner)
    (rightOwner := SemanticResult137768.owner)
    (leftResult := 143120) (rightResult := 137768)
    (leftActual := SemanticResult143120.actual selector witness)
    (rightActual := SemanticResult137768.actual selector witness)
    (leftRaw := SemanticResult143120.rawTerms)
    (rightRaw := SemanticResult137768.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 354089550391067611616654269349888)
    (rightMaximum := 32192146870060392302605751287808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143121) (rightBinding := 143122)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69629⟩) (rightExpression := ⟨30797⟩)
    (transferEvent := 143123) (summaryTransferEvent := 143124)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143120.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult137768.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143125

namespace SemanticResult143130
def owner : Owner := ⟨.program ⟨257⟩, ⟨69631⟩⟩
def rawTerms : List Term := Proof.Events559.exact143130RawTerms
def summary : Bound := (.finite 418474237032079770976347551432704)
def resultEvent : Nat := 143130
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult143130.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult143125.owner)
    (rightOwner := SemanticResult137286.owner)
    (leftResult := 143125) (rightResult := 137286)
    (leftActual := SemanticResult143125.actual selector witness)
    (rightActual := SemanticResult137286.actual selector witness)
    (leftRaw := SemanticResult143125.rawTerms)
    (rightRaw := SemanticResult137286.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 386281697261128003919260020637696)
    (rightMaximum := 32192539770951767057087530795008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 143126) (rightBinding := 143127)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨69630⟩) (rightExpression := ⟨36457⟩)
    (transferEvent := 143128) (summaryTransferEvent := 143129)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult143125.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult137286.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult143130

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
