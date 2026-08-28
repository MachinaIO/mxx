import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1153
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard127
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1055
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1142
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1143
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1144
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1146
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1147
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1148
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1150
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1151
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1152

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult163131
def owner : Owner := ⟨.program ⟨257⟩, ⟨17673⟩⟩
def rawTerms : List Term := Proof.Events637.exact163131RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 163131
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163131.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge163128.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163124.owner)
    (rightOwner := SemanticResult162946.owner)
    (leftResult := 163124) (rightResult := 162946)
    (leftActual := SemanticResult163124.actual selector witness)
    (rightActual := SemanticResult162946.actual selector witness)
    (leftRaw := SemanticResult163124.rawTerms)
    (rightRaw := SemanticResult162946.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163125) (rightBinding := 163126)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16535⟩) (rightExpression := ⟨17672⟩)
    (coefficientTransfer := 163127) (summaryTransfer := 163130)
    (base := LeftOperatorMerge163128.base)
    (reconstruction := LeftOperatorMerge163128.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163124.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult162946.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge163128.operationAgreement
  · rfl
  · decide
end SemanticResult163131

namespace SemanticResult163141
def owner : Owner := ⟨.program ⟨257⟩, ⟨17674⟩⟩
def rawTerms : List Term := Proof.Events637.exact163141RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529920)
def resultEvent : Nat := 163141
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163141.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨32188807212483706889510625476608, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge163137.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge163137.frameStart)
    (owner := owner) (leftOwner := SemanticResult163131.owner)
    (rightOwner := SemanticResult15882.owner)
    (leftResult := 163131) (rightResult := 15882)
    (leftActual := SemanticResult163131.actual selector witness)
    (rightActual := SemanticResult15882.actual selector witness)
    (leftRaw := SemanticResult163131.rawTerms)
    (rightRaw := SemanticResult15882.rawTerms)
    (working := LeftOperatorMerge163137.working)
    (leftBinding := 163132) (rightBinding := 163133)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17673⟩) (rightExpression := ⟨7172⟩)
    (coefficientTransfer := 163134) (summaryTransfer := 163136)
    (rightCoefficientProducer := 15881)
    (rightSummaryTransfer := 163135)
    (leftMaximum := ⟨32188807212483706889510625476608, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge163137.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15881.actual selector witness)
    (summaryMagnitude := LeftBound163136.actual selector witness)
    (reconstruction := LeftOperatorMerge163137.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163131.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15882.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15881.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15881.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge163137.operationAgreement
  · exact LeftBound163136.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge163137.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 163139 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge163137.working
    [{ coefficient := (-1), key := LeftRelationMerge163139.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge163139.frameStart
      LeftRelationMerge163139.owner (.relation 163139) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge163139.deltas
    rows := LeftRelationMerge163139.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge163137.working LeftRelationMerge163139.source
        (relationContext LeftRelationMerge163139.source
          LeftRelationMerge163139.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge163137.working, LeftRelationMerge163139.deltas,
    LeftRelationMerge163139.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 163139)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨17674⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge163137.working) (working := relationWorking0)
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
end SemanticResult163141

namespace SemanticResult163146
def owner : Owner := ⟨.program ⟨257⟩, ⟨7076⟩⟩
def rawTerms : List Term := Proof.Events637.exact163146RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 163146
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163146.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge163145.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge163145.frameStart)
    (transferEvent := 163144) (owner := owner)
    (leftResult := 723) (rightResult := 149028)
    (working := LeftOperatorMerge163145.working)
    (reconstruction := LeftOperatorMerge163145.reconstruction)
    (leftReference := .predecessor 0 163142 .coefficient) (rightReference := .predecessor 1 163143 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult149028.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge163145.operationAgreement
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
end SemanticResult163146

namespace SemanticResult163151
def owner : Owner := ⟨.program ⟨257⟩, ⟨8256⟩⟩
def rawTerms : List Term := Proof.Events637.exact163151RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 163151
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163151.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge163150.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge163150.frameStart)
    (transferEvent := 163149) (owner := owner)
    (leftResult := 148898) (rightResult := 15896)
    (working := LeftOperatorMerge163150.working)
    (reconstruction := LeftOperatorMerge163150.reconstruction)
    (leftReference := .predecessor 0 163147 .coefficient) (rightReference := .predecessor 1 163148 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult148898.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15896.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge163150.operationAgreement
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
end SemanticResult163151

namespace SemanticResult163155
def owner : Owner := ⟨.program ⟨257⟩, ⟨9353⟩⟩
def rawTerms : List Term := Proof.Events637.exact163155RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 163155
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163155.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 163152) (rightBinding := 163153)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨8256⟩) (rightExpression := ⟨7076⟩)
    (transferEvent := 163154)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163151.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult163146.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163155

namespace SemanticResult163161
def owner : Owner := ⟨.program ⟨257⟩, ⟨9354⟩⟩
def rawTerms : List Term := Proof.Events637.exact163161RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 163161
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163161.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 163158) (survivorTransfer := 163159)
    (survivorEvent := 163160) (resultEvent := resultEvent)
    (rightCoefficientProducer := 31515)
    (owner := owner) (leftOwner := SemanticResult163155.owner)
    (rightOwner := SemanticResult31516.owner)
    (leftResult := 163155) (rightResult := 31516)
    (leftBinding := 163156) (rightBinding := 163157)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9353⟩) (rightExpression := ⟨118⟩)
    (leftActual := SemanticResult163155.actual selector witness)
    (rightActual := SemanticResult31516.actual selector witness)
    (leftRaw := SemanticResult163155.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound31515.actual selector witness)
    (survivorMagnitude := LeftBound163159.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163155.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)
  · exact LeftBound163159.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult163161

namespace SemanticResult163168
def owner : Owner := ⟨.program ⟨257⟩, ⟨9468⟩⟩
def rawTerms : List Term := Proof.Events637.exact163168RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 163168
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163168.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge163165.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163161.owner)
    (rightOwner := SemanticResult163161.owner)
    (leftResult := 163161) (rightResult := 163161)
    (leftActual := SemanticResult163161.actual selector witness)
    (rightActual := SemanticResult163161.actual selector witness)
    (leftRaw := SemanticResult163161.rawTerms)
    (rightRaw := SemanticResult163161.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163162) (rightBinding := 163163)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9354⟩) (rightExpression := ⟨9354⟩)
    (coefficientTransfer := 163164) (summaryTransfer := 163167)
    (base := LeftOperatorMerge163165.base)
    (reconstruction := LeftOperatorMerge163165.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163161.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult163161.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge163165.operationAgreement
  · rfl
  · decide
end SemanticResult163168

namespace SemanticResult163173
def owner : Owner := ⟨.program ⟨257⟩, ⟨17675⟩⟩
def rawTerms : List Term := Proof.Events637.exact163173RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 163173
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163173.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163168.owner)
    (rightOwner := SemanticResult163141.owner)
    (leftResult := 163168) (rightResult := 163141)
    (leftActual := SemanticResult163168.actual selector witness)
    (rightActual := SemanticResult163141.actual selector witness)
    (leftRaw := SemanticResult163168.rawTerms)
    (rightRaw := SemanticResult163141.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163169) (rightBinding := 163170)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9468⟩) (rightExpression := ⟨17674⟩)
    (transferEvent := 163171) (summaryTransferEvent := 163172)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163168.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult163141.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163173

namespace SemanticResult163178
def owner : Owner := ⟨.program ⟨257⟩, ⟨20557⟩⟩
def rawTerms : List Term := Proof.Events637.exact163178RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 163178
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163178.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163173.owner)
    (rightOwner := SemanticResult162929.owner)
    (leftResult := 163173) (rightResult := 162929)
    (leftActual := SemanticResult163173.actual selector witness)
    (rightActual := SemanticResult162929.actual selector witness)
    (leftRaw := SemanticResult163173.rawTerms)
    (rightRaw := SemanticResult162929.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163174) (rightBinding := 163175)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17675⟩) (rightExpression := ⟨20556⟩)
    (transferEvent := 163176) (summaryTransferEvent := 163177)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163173.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult162929.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163178

namespace SemanticResult163183
def owner : Owner := ⟨.program ⟨257⟩, ⟨23777⟩⟩
def rawTerms : List Term := Proof.Events637.exact163183RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 163183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163183.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163178.owner)
    (rightOwner := SemanticResult162717.owner)
    (leftResult := 163178) (rightResult := 162717)
    (leftActual := SemanticResult163178.actual selector witness)
    (rightActual := SemanticResult162717.actual selector witness)
    (leftRaw := SemanticResult163178.rawTerms)
    (rightRaw := SemanticResult162717.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163179) (rightBinding := 163180)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20557⟩) (rightExpression := ⟨23776⟩)
    (transferEvent := 163181) (summaryTransferEvent := 163182)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163178.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult162717.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163183

namespace SemanticResult163188
def owner : Owner := ⟨.program ⟨257⟩, ⟨33797⟩⟩
def rawTerms : List Term := Proof.Events637.exact163188RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 163188
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163188.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163183.owner)
    (rightOwner := SemanticResult162505.owner)
    (leftResult := 163183) (rightResult := 162505)
    (leftActual := SemanticResult163183.actual selector witness)
    (rightActual := SemanticResult162505.actual selector witness)
    (leftRaw := SemanticResult163183.rawTerms)
    (rightRaw := SemanticResult162505.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163184) (rightBinding := 163185)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨23777⟩) (rightExpression := ⟨33796⟩)
    (transferEvent := 163186) (summaryTransferEvent := 163187)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163183.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult162505.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163188

namespace SemanticResult163193
def owner : Owner := ⟨.program ⟨257⟩, ⟨52857⟩⟩
def rawTerms : List Term := Proof.Events637.exact163193RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 163193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163193.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163188.owner)
    (rightOwner := SemanticResult162293.owner)
    (leftResult := 163188) (rightResult := 162293)
    (leftActual := SemanticResult163188.actual selector witness)
    (rightActual := SemanticResult162293.actual selector witness)
    (leftRaw := SemanticResult163188.rawTerms)
    (rightRaw := SemanticResult162293.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163189) (rightBinding := 163190)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨33797⟩) (rightExpression := ⟨52856⟩)
    (transferEvent := 163191) (summaryTransferEvent := 163192)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163188.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult162293.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163193

namespace SemanticResult163198
def owner : Owner := ⟨.program ⟨257⟩, ⟨55837⟩⟩
def rawTerms : List Term := Proof.Events637.exact163198RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 163198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163198.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163193.owner)
    (rightOwner := SemanticResult162081.owner)
    (leftResult := 163193) (rightResult := 162081)
    (leftActual := SemanticResult163193.actual selector witness)
    (rightActual := SemanticResult162081.actual selector witness)
    (leftRaw := SemanticResult163193.rawTerms)
    (rightRaw := SemanticResult162081.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163194) (rightBinding := 163195)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨52857⟩) (rightExpression := ⟨55836⟩)
    (transferEvent := 163196) (summaryTransferEvent := 163197)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163193.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult162081.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163198

namespace SemanticResult163203
def owner : Owner := ⟨.program ⟨257⟩, ⟨58817⟩⟩
def rawTerms : List Term := Proof.Events637.exact163203RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 163203
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163203.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163198.owner)
    (rightOwner := SemanticResult161869.owner)
    (leftResult := 163198) (rightResult := 161869)
    (leftActual := SemanticResult163198.actual selector witness)
    (rightActual := SemanticResult161869.actual selector witness)
    (leftRaw := SemanticResult163198.rawTerms)
    (rightRaw := SemanticResult161869.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163199) (rightBinding := 163200)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨55837⟩) (rightExpression := ⟨58816⟩)
    (transferEvent := 163201) (summaryTransferEvent := 163202)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163198.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult161869.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163203

namespace SemanticResult163208
def owner : Owner := ⟨.program ⟨257⟩, ⟨61797⟩⟩
def rawTerms : List Term := Proof.Events637.exact163208RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 163208
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163208.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163203.owner)
    (rightOwner := SemanticResult161657.owner)
    (leftResult := 163203) (rightResult := 161657)
    (leftActual := SemanticResult163203.actual selector witness)
    (rightActual := SemanticResult161657.actual selector witness)
    (leftRaw := SemanticResult163203.rawTerms)
    (rightRaw := SemanticResult161657.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163204) (rightBinding := 163205)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨58817⟩) (rightExpression := ⟨61796⟩)
    (transferEvent := 163206) (summaryTransferEvent := 163207)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163203.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult161657.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163208

namespace SemanticResult163213
def owner : Owner := ⟨.program ⟨257⟩, ⟨64777⟩⟩
def rawTerms : List Term := Proof.Events637.exact163213RawTerms
def summary : Bound := (.finite 3110701272581949232038858886277070355169332)
def resultEvent : Nat := 163213
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult163213.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult163208.owner)
    (rightOwner := SemanticResult161445.owner)
    (leftResult := 163208) (rightResult := 161445)
    (leftActual := SemanticResult163208.actual selector witness)
    (rightActual := SemanticResult161445.actual selector witness)
    (leftRaw := SemanticResult163208.rawTerms)
    (rightRaw := SemanticResult161445.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2765055493188795324243372926469393465999412)
    (rightMaximum := 345645779393153907795485959807676889169920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 163209) (rightBinding := 163210)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨61797⟩) (rightExpression := ⟨64776⟩)
    (transferEvent := 163211) (summaryTransferEvent := 163212)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult163208.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult161445.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult163213

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
