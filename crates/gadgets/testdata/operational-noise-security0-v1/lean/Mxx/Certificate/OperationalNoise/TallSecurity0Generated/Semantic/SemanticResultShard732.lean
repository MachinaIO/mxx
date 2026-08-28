import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard732
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard667
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard710
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard714
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard717
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard721
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard725
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard728
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard731

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult102126
def owner : Owner := ⟨.program ⟨214⟩, ⟨6690⟩⟩
def rawTerms : List Term := Proof.Events398.exact102126RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 102126
def producerEvent : Nat := 102125
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102126.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 102068, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult102126

namespace SemanticResult102130
def owner : Owner := ⟨.program ⟨214⟩, ⟨14827⟩⟩
def rawTerms : List Term := Proof.Events398.exact102130RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 102130
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102130.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 102127) (rightBinding := 102128)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6690⟩) (rightExpression := ⟨14826⟩)
    (transferEvent := 102129)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102126.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102123.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult102130

namespace SemanticResult102138
def owner : Owner := ⟨.program ⟨214⟩, ⟨26327⟩⟩
def rawTerms : List Term := Proof.Events398.exact102138RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 102138
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102138.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge102134.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge102134.frameStart)
    (transferEvent := 102133) (owner := owner)
    (leftResult := 102130) (rightResult := 102107)
    (working := LeftOperatorMerge102134.working)
    (reconstruction := LeftOperatorMerge102134.reconstruction)
    (leftReference := .predecessor 0 102131 .coefficient) (rightReference := .predecessor 1 102132 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult102130.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102107.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge102134.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 102136 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge102134.working
    [{ coefficient := (-1), key := LeftRelationMerge102136.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge102136.frameStart
      LeftRelationMerge102136.owner (.relation 102136) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge102136.deltas
    rows := LeftRelationMerge102136.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge102134.working LeftRelationMerge102136.source
        (relationContext LeftRelationMerge102136.source
          LeftRelationMerge102136.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge102134.working, LeftRelationMerge102136.deltas,
    LeftRelationMerge102136.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 102136)
    (frameStart := 102068) (owner := ⟨.program ⟨214⟩, ⟨26327⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨14782⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge102134.working) (working := relationWorking0)
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult102138

namespace SemanticResult102141
def owner : Owner := ⟨.program ⟨214⟩, ⟨15258⟩⟩
def rawTerms : List Term := Proof.Events398.exact102141RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 102141
def producerEvent : Nat := 102140
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102141.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 102068, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult102141

namespace SemanticResult102146
def owner : Owner := ⟨.program ⟨214⟩, ⟨15259⟩⟩
def rawTerms : List Term := Proof.Events399.exact102146RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 102146
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102146.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge102145.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge102145.frameStart)
    (transferEvent := 102144) (owner := owner)
    (leftResult := 102118) (rightResult := 102141)
    (working := LeftOperatorMerge102145.working)
    (reconstruction := LeftOperatorMerge102145.reconstruction)
    (leftReference := .predecessor 0 102142 .coefficient) (rightReference := .predecessor 1 102143 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult102118.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102141.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge102145.operationAgreement
  · decide

theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult102146

namespace SemanticResult102149
def owner : Owner := ⟨.program ⟨214⟩, ⟨6709⟩⟩
def rawTerms : List Term := Proof.Events399.exact102149RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 102149
def producerEvent : Nat := 102148
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102149.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 102068, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult102149

namespace SemanticResult102153
def owner : Owner := ⟨.program ⟨214⟩, ⟨15260⟩⟩
def rawTerms : List Term := Proof.Events399.exact102153RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 102153
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102153.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 102150) (rightBinding := 102151)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6709⟩) (rightExpression := ⟨15259⟩)
    (transferEvent := 102152)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102149.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102146.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult102153

namespace SemanticResult102157
def owner : Owner := ⟨.program ⟨214⟩, ⟨26330⟩⟩
def rawTerms : List Term := Proof.Events399.exact102157RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 102157
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102157.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 102154) (rightBinding := 102155)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨15260⟩) (rightExpression := ⟨26327⟩)
    (transferEvent := 102156)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102153.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102138.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult102157

namespace SemanticResult102166
def owner : Owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩
def rawTerms : List Term := Proof.Events399.exact102166RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 102166
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102166.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge102025.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge102025.frameStart)
    (owner := owner) (leftOwner := SemanticResult94462.owner)
    (rightOwner := SemanticResult102019.owner)
    (leftResult := 94462) (rightResult := 102019)
    (leftActual := SemanticResult94462.actual selector witness)
    (rightActual := SemanticResult102019.actual selector witness)
    (leftRaw := SemanticResult94462.rawTerms)
    (rightRaw := SemanticResult102019.rawTerms)
    (working := LeftOperatorMerge102025.working)
    (leftBinding := 102020) (rightBinding := 102021)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5509⟩) (rightExpression := ⟨20383⟩)
    (coefficientTransfer := 102022) (summaryTransfer := 102024)
    (rightCoefficientProducer := 102018)
    (rightSummaryTransfer := 102023)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge102025.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound102018.actual selector witness)
    (summaryMagnitude := LeftBound102024.actual selector witness)
    (reconstruction := LeftOperatorMerge102025.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult94462.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102019.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102018.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound102018.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge102025.operationAgreement
  · exact LeftBound102024.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge102025.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 102161 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26326⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14782⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23712⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15258⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge102025.working
    [{ coefficient := (1), key := LeftRelationMerge102161.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge102161.frameStart
      LeftRelationMerge102161.owner (.relation 102161) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge102161.deltas
    rows := LeftRelationMerge102161.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge102025.working LeftRelationMerge102161.source
        (relationContext LeftRelationMerge102161.source
          LeftRelationMerge102161.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge102025.working, LeftRelationMerge102161.deltas,
    LeftRelationMerge102161.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 102161)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨20384⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20381⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge102025.working) (working := relationWorking0)
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
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim0 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult102166

namespace SemanticResult102173
def owner : Owner := ⟨.program ⟨214⟩, ⟨26329⟩⟩
def rawTerms : List Term := Proof.Events399.exact102173RawTerms
def summary : Bound := (.finite 1291889174379421642752)
def resultEvent : Nat := 102173
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102173.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge102170.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult102166.owner)
    (rightOwner := SemanticResult102012.owner)
    (leftResult := 102166) (rightResult := 102012)
    (leftActual := SemanticResult102166.actual selector witness)
    (rightActual := SemanticResult102012.actual selector witness)
    (leftRaw := SemanticResult102166.rawTerms)
    (rightRaw := SemanticResult102012.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1291889172568118132736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 102167) (rightBinding := 102168)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20384⟩) (rightExpression := ⟨26328⟩)
    (coefficientTransfer := 102169) (summaryTransfer := 102172)
    (base := LeftOperatorMerge102170.base)
    (reconstruction := LeftOperatorMerge102170.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102166.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult102012.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge102170.operationAgreement
  · rfl
  · decide
end SemanticResult102173

namespace SemanticResult102178
def owner : Owner := ⟨.program ⟨214⟩, ⟨26533⟩⟩
def rawTerms : List Term := Proof.Events399.exact102178RawTerms
def summary : Bound := (.finite 2583789554981353578496)
def resultEvent : Nat := 102178
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102178.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult102173.owner)
    (rightOwner := SemanticResult101739.owner)
    (leftResult := 102173) (rightResult := 101739)
    (leftActual := SemanticResult102173.actual selector witness)
    (rightActual := SemanticResult101739.actual selector witness)
    (leftRaw := SemanticResult102173.rawTerms)
    (rightRaw := SemanticResult101739.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1291889174379421642752)
    (rightMaximum := 1291900380601931935744) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 102174) (rightBinding := 102175)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26329⟩) (rightExpression := ⟨26532⟩)
    (transferEvent := 102176) (summaryTransferEvent := 102177)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102173.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult101739.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult102178

namespace SemanticResult102183
def owner : Owner := ⟨.program ⟨214⟩, ⟨26750⟩⟩
def rawTerms : List Term := Proof.Events399.exact102183RawTerms
def summary : Bound := (.finite 3875701141805795807232)
def resultEvent : Nat := 102183
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102183.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult102178.owner)
    (rightOwner := SemanticResult101305.owner)
    (leftResult := 102178) (rightResult := 101305)
    (leftActual := SemanticResult102178.actual selector witness)
    (rightActual := SemanticResult101305.actual selector witness)
    (leftRaw := SemanticResult102178.rawTerms)
    (rightRaw := SemanticResult101305.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2583789554981353578496)
    (rightMaximum := 1291911586824442228736) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 102179) (rightBinding := 102180)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26533⟩) (rightExpression := ⟨26749⟩)
    (transferEvent := 102181) (summaryTransferEvent := 102182)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102178.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult101305.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult102183

namespace SemanticResult102188
def owner : Owner := ⟨.program ⟨214⟩, ⟨26967⟩⟩
def rawTerms : List Term := Proof.Events399.exact102188RawTerms
def summary : Bound := (.finite 5167635141075258621952)
def resultEvent : Nat := 102188
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102188.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult102183.owner)
    (rightOwner := SemanticResult100871.owner)
    (leftResult := 102183) (rightResult := 100871)
    (leftActual := SemanticResult102183.actual selector witness)
    (rightActual := SemanticResult100871.actual selector witness)
    (leftRaw := SemanticResult102183.rawTerms)
    (rightRaw := SemanticResult100871.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 3875701141805795807232)
    (rightMaximum := 1291933999269462814720) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 102184) (rightBinding := 102185)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26750⟩) (rightExpression := ⟨26966⟩)
    (transferEvent := 102186) (summaryTransferEvent := 102187)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102183.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100871.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult102188

namespace SemanticResult102193
def owner : Owner := ⟨.program ⟨214⟩, ⟨27184⟩⟩
def rawTerms : List Term := Proof.Events399.exact102193RawTerms
def summary : Bound := (.finite 6459613965234762608640)
def resultEvent : Nat := 102193
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102193.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult102188.owner)
    (rightOwner := SemanticResult100437.owner)
    (leftResult := 102188) (rightResult := 100437)
    (leftActual := SemanticResult102188.actual selector witness)
    (rightActual := SemanticResult100437.actual selector witness)
    (leftRaw := SemanticResult102188.rawTerms)
    (rightRaw := SemanticResult100437.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 5167635141075258621952)
    (rightMaximum := 1291978824159503986688) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 102189) (rightBinding := 102190)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨26967⟩) (rightExpression := ⟨27183⟩)
    (transferEvent := 102191) (summaryTransferEvent := 102192)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102188.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100437.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult102193

namespace SemanticResult102198
def owner : Owner := ⟨.program ⟨214⟩, ⟨27401⟩⟩
def rawTerms : List Term := Proof.Events399.exact102198RawTerms
def summary : Bound := (.finite 7751615201839287181312)
def resultEvent : Nat := 102198
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102198.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult102193.owner)
    (rightOwner := SemanticResult100003.owner)
    (leftResult := 102193) (rightResult := 100003)
    (leftActual := SemanticResult102193.actual selector witness)
    (rightActual := SemanticResult100003.actual selector witness)
    (leftRaw := SemanticResult102193.rawTerms)
    (rightRaw := SemanticResult100003.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 6459613965234762608640)
    (rightMaximum := 1292001236604524572672) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 102194) (rightBinding := 102195)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27184⟩) (rightExpression := ⟨27400⟩)
    (transferEvent := 102196) (summaryTransferEvent := 102197)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102193.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult100003.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult102198

namespace SemanticResult102203
def owner : Owner := ⟨.program ⟨214⟩, ⟨27618⟩⟩
def rawTerms : List Term := Proof.Events399.exact102203RawTerms
def summary : Bound := (.finite 9043661263333852925952)
def resultEvent : Nat := 102203
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult102203.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult102198.owner)
    (rightOwner := SemanticResult99569.owner)
    (leftResult := 102198) (rightResult := 99569)
    (leftActual := SemanticResult102198.actual selector witness)
    (rightActual := SemanticResult99569.actual selector witness)
    (leftRaw := SemanticResult102198.rawTerms)
    (rightRaw := SemanticResult99569.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 7751615201839287181312)
    (rightMaximum := 1292046061494565744640) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 102199) (rightBinding := 102200)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨27401⟩) (rightExpression := ⟨27617⟩)
    (transferEvent := 102201) (summaryTransferEvent := 102202)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult102198.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99569.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult102203

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
