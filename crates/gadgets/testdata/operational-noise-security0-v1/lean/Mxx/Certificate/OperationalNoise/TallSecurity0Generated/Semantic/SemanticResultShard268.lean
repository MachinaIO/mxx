import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard268
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard013
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard265
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard267

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult36498
def owner : Owner := ⟨.program ⟨214⟩, ⟨18178⟩⟩
def rawTerms : List Term := Proof.Events142.exact36498RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36498
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36498.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 36495) (rightBinding := 36496)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6743⟩) (rightExpression := ⟨18177⟩)
    (transferEvent := 36497)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36494.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36491.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult36498

namespace SemanticResult36502
def owner : Owner := ⟨.program ⟨214⟩, ⟨30169⟩⟩
def rawTerms : List Term := Proof.Events142.exact36502RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36502
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36502.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 36499) (rightBinding := 36500)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18178⟩) (rightExpression := ⟨30162⟩)
    (transferEvent := 36501)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36498.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36483.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult36502

namespace SemanticResult36511
def owner : Owner := ⟨.program ⟨214⟩, ⟨22851⟩⟩
def rawTerms : List Term := Proof.Events142.exact36511RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 36511
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36511.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36346.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge36346.frameStart)
    (owner := owner) (leftOwner := SemanticResult36137.owner)
    (rightOwner := SemanticResult36340.owner)
    (leftResult := 36137) (rightResult := 36340)
    (leftActual := SemanticResult36137.actual selector witness)
    (rightActual := SemanticResult36340.actual selector witness)
    (leftRaw := SemanticResult36137.rawTerms)
    (rightRaw := SemanticResult36340.rawTerms)
    (working := LeftOperatorMerge36346.working)
    (leftBinding := 36341) (rightBinding := 36342)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5553⟩) (rightExpression := ⟨22850⟩)
    (coefficientTransfer := 36343) (summaryTransfer := 36345)
    (rightCoefficientProducer := 36339)
    (rightSummaryTransfer := 36344)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge36346.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound36339.actual selector witness)
    (summaryMagnitude := LeftBound36345.actual selector witness)
    (reconstruction := LeftOperatorMerge36346.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36137.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36340.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36339.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound36339.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge36346.operationAgreement
  · exact LeftBound36345.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36346.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 36506 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30161⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24798⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18176⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30161⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24798⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18176⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge36346.working
    [{ coefficient := (1), key := LeftRelationMerge36506.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge36506.frameStart
      LeftRelationMerge36506.owner (.relation 36506) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge36506.deltas
    rows := LeftRelationMerge36506.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge36346.working LeftRelationMerge36506.source
        (relationContext LeftRelationMerge36506.source
          LeftRelationMerge36506.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge36346.working, LeftRelationMerge36506.deltas,
    LeftRelationMerge36506.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 36506)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨22851⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22848⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22848⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge36346.working) (working := relationWorking0)
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
end SemanticResult36511

namespace SemanticResult36518
def owner : Owner := ⟨.program ⟨214⟩, ⟨30164⟩⟩
def rawTerms : List Term := Proof.Events142.exact36518RawTerms
def summary : Bound := (.finite 1292539135285018636288)
def resultEvent : Nat := 36518
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36518.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge36515.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult36511.owner)
    (rightOwner := SemanticResult36333.owner)
    (leftResult := 36511) (rightResult := 36333)
    (leftActual := SemanticResult36511.actual selector witness)
    (rightActual := SemanticResult36333.actual selector witness)
    (leftRaw := SemanticResult36511.rawTerms)
    (rightRaw := SemanticResult36333.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292539133473715126272) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 36512) (rightBinding := 36513)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22851⟩) (rightExpression := ⟨30163⟩)
    (coefficientTransfer := 36514) (summaryTransfer := 36517)
    (base := LeftOperatorMerge36515.base)
    (reconstruction := LeftOperatorMerge36515.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36511.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36333.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36515.operationAgreement
  · rfl
  · decide
end SemanticResult36518

namespace SemanticResult36525
def owner : Owner := ⟨.program ⟨214⟩, ⟨24735⟩⟩
def rawTerms : List Term := Proof.Events142.exact36525RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36525
def producerEvent : Nat := 36524
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36525.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36525

namespace SemanticResult36528
def owner : Owner := ⟨.program ⟨214⟩, ⟨29845⟩⟩
def rawTerms : List Term := Proof.Events142.exact36528RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36528
def producerEvent : Nat := 36527
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36528.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36528

namespace SemanticResult36535
def owner : Owner := ⟨.program ⟨214⟩, ⟨23378⟩⟩
def rawTerms : List Term := Proof.Events142.exact36535RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36535
def producerEvent : Nat := 36534
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36535.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36535

namespace SemanticResult36538
def owner : Owner := ⟨.program ⟨214⟩, ⟨25691⟩⟩
def rawTerms : List Term := Proof.Events142.exact36538RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36538
def producerEvent : Nat := 36537
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36538.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 0, .finite 8192, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult36538

namespace SemanticResult36543
def owner : Owner := ⟨.program ⟨214⟩, ⟨13173⟩⟩
def rawTerms : List Term := Proof.Events142.exact36543RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36543
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36543.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36542.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36542.frameStart)
    (transferEvent := 36541) (owner := owner)
    (leftResult := 1613) (rightResult := 36045)
    (working := LeftOperatorMerge36542.working)
    (reconstruction := LeftOperatorMerge36542.reconstruction)
    (leftReference := .predecessor 0 36539 .coefficient) (rightReference := .predecessor 1 36540 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1613.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36542.operationAgreement
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
end SemanticResult36543

namespace SemanticResult36548
def owner : Owner := ⟨.program ⟨214⟩, ⟨7321⟩⟩
def rawTerms : List Term := Proof.Events142.exact36548RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36548
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36548.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36547.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36547.frameStart)
    (transferEvent := 36546) (owner := owner)
    (leftResult := 35915) (rightResult := 6973)
    (working := LeftOperatorMerge36547.working)
    (reconstruction := LeftOperatorMerge36547.reconstruction)
    (leftReference := .predecessor 0 36544 .coefficient) (rightReference := .predecessor 1 36545 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6973.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36547.operationAgreement
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
end SemanticResult36548

namespace SemanticResult36552
def owner : Owner := ⟨.program ⟨214⟩, ⟨13174⟩⟩
def rawTerms : List Term := Proof.Events142.exact36552RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36552
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36552.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 36549) (rightBinding := 36550)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7321⟩) (rightExpression := ⟨13173⟩)
    (transferEvent := 36551)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36548.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36543.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult36552

namespace SemanticResult36558
def owner : Owner := ⟨.program ⟨214⟩, ⟨13175⟩⟩
def rawTerms : List Term := Proof.Events142.exact36558RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 36558
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36558.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 36555) (survivorTransfer := 36556)
    (survivorEvent := 36557) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6964)
    (owner := owner) (leftOwner := SemanticResult36552.owner)
    (rightOwner := SemanticResult6965.owner)
    (leftResult := 36552) (rightResult := 6965)
    (leftBinding := 36553) (rightBinding := 36554)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13174⟩) (rightExpression := ⟨103⟩)
    (leftActual := SemanticResult36552.actual selector witness)
    (rightActual := SemanticResult6965.actual selector witness)
    (leftRaw := SemanticResult36552.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6964.actual selector witness)
    (survivorMagnitude := LeftBound36556.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36552.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6965.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)
  · exact LeftBound36556.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult36558

namespace SemanticResult36566
def owner : Owner := ⟨.program ⟨214⟩, ⟨13176⟩⟩
def rawTerms : List Term := Proof.Events142.exact36566RawTerms
def summary : Bound := (.finite 48256)
def resultEvent : Nat := 36566
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36566.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨58, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36564.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge36564.frameStart)
    (owner := owner) (leftOwner := SemanticResult36558.owner)
    (rightOwner := SemanticResult1616.owner)
    (leftResult := 36558) (rightResult := 1616)
    (leftActual := SemanticResult36558.actual selector witness)
    (rightActual := SemanticResult1616.actual selector witness)
    (leftRaw := SemanticResult36558.rawTerms)
    (rightRaw := SemanticResult1616.rawTerms)
    (working := LeftOperatorMerge36564.working)
    (leftBinding := 36559) (rightBinding := 36560)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13175⟩) (rightExpression := ⟨10250⟩)
    (coefficientTransfer := 36561) (summaryTransfer := 36563)
    (rightCoefficientProducer := 1615)
    (rightSummaryTransfer := 36562)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨58, by decide⟩)
    (rightRecordedMaximum := 58)
    (rightSummaryMaximum := ⟨58, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge36564.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1615.actual selector witness)
    (summaryMagnitude := LeftBound36563.actual selector witness)
    (reconstruction := LeftOperatorMerge36564.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36558.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1616.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1615.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1615.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge36564.operationAgreement
  · exact LeftBound36563.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36564.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (mergeClaim selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult36566

namespace SemanticResult36571
def owner : Owner := ⟨.program ⟨214⟩, ⟨10251⟩⟩
def rawTerms : List Term := Proof.Events142.exact36571RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36571
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36571.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36570.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36570.frameStart)
    (transferEvent := 36569) (owner := owner)
    (leftResult := 1616) (rightResult := 36045)
    (working := LeftOperatorMerge36570.working)
    (reconstruction := LeftOperatorMerge36570.reconstruction)
    (leftReference := .predecessor 0 36567 .coefficient) (rightReference := .predecessor 1 36568 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1616.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36570.operationAgreement
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
end SemanticResult36571

namespace SemanticResult36576
def owner : Owner := ⟨.program ⟨214⟩, ⟨7301⟩⟩
def rawTerms : List Term := Proof.Events142.exact36576RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36576
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36576.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge36575.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge36575.frameStart)
    (transferEvent := 36574) (owner := owner)
    (leftResult := 35915) (rightResult := 7014)
    (working := LeftOperatorMerge36575.working)
    (reconstruction := LeftOperatorMerge36575.reconstruction)
    (leftReference := .predecessor 0 36572 .coefficient) (rightReference := .predecessor 1 36573 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult7014.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge36575.operationAgreement
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
end SemanticResult36576

namespace SemanticResult36580
def owner : Owner := ⟨.program ⟨214⟩, ⟨10252⟩⟩
def rawTerms : List Term := Proof.Events142.exact36580RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 36580
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult36580.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 36577) (rightBinding := 36578)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7301⟩) (rightExpression := ⟨10251⟩)
    (transferEvent := 36579)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult36576.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36571.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult36580

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
