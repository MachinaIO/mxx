import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard469
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard465
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard466
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard468

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult65741
def owner : Owner := ⟨.program ⟨214⟩, ⟨18168⟩⟩
def rawTerms : List Term := Proof.Events256.exact65741RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65741
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65741.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65740.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge65740.frameStart)
    (transferEvent := 65739) (owner := owner)
    (leftResult := 65713) (rightResult := 65736)
    (working := LeftOperatorMerge65740.working)
    (reconstruction := LeftOperatorMerge65740.reconstruction)
    (leftReference := .predecessor 0 65737 .coefficient) (rightReference := .predecessor 1 65738 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65713.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65736.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65740.operationAgreement
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
end SemanticResult65741

namespace SemanticResult65744
def owner : Owner := ⟨.program ⟨214⟩, ⟨6743⟩⟩
def rawTerms : List Term := Proof.Events256.exact65744RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65744
def producerEvent : Nat := 65743
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65744.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 65651, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult65744

namespace SemanticResult65748
def owner : Owner := ⟨.program ⟨214⟩, ⟨18169⟩⟩
def rawTerms : List Term := Proof.Events256.exact65748RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65748
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65748.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 65745) (rightBinding := 65746)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6743⟩) (rightExpression := ⟨18168⟩)
    (transferEvent := 65747)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65744.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65741.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult65748

namespace SemanticResult65752
def owner : Owner := ⟨.program ⟨214⟩, ⟨30103⟩⟩
def rawTerms : List Term := Proof.Events256.exact65752RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65752
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65752.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 65749) (rightBinding := 65750)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18169⟩) (rightExpression := ⟨30096⟩)
    (transferEvent := 65751)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65748.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65733.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult65752

namespace SemanticResult65761
def owner : Owner := ⟨.program ⟨214⟩, ⟨22839⟩⟩
def rawTerms : List Term := Proof.Events256.exact65761RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 65761
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65761.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65596.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge65596.frameStart)
    (owner := owner) (leftOwner := SemanticResult65387.owner)
    (rightOwner := SemanticResult65590.owner)
    (leftResult := 65387) (rightResult := 65590)
    (leftActual := SemanticResult65387.actual selector witness)
    (rightActual := SemanticResult65590.actual selector witness)
    (leftRaw := SemanticResult65387.rawTerms)
    (rightRaw := SemanticResult65590.rawTerms)
    (working := LeftOperatorMerge65596.working)
    (leftBinding := 65591) (rightBinding := 65592)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5535⟩) (rightExpression := ⟨22838⟩)
    (coefficientTransfer := 65593) (summaryTransfer := 65595)
    (rightCoefficientProducer := 65589)
    (rightSummaryTransfer := 65594)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge65596.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound65589.actual selector witness)
    (summaryMagnitude := LeftBound65595.actual selector witness)
    (reconstruction := LeftOperatorMerge65596.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65387.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65590.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65589.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound65589.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge65596.operationAgreement
  · exact LeftBound65595.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65596.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 65756 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30095⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17007⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24789⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30095⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17007⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24789⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge65596.working
    [{ coefficient := (1), key := LeftRelationMerge65756.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge65756.frameStart
      LeftRelationMerge65756.owner (.relation 65756) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge65756.deltas
    rows := LeftRelationMerge65756.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge65596.working LeftRelationMerge65756.source
        (relationContext LeftRelationMerge65756.source
          LeftRelationMerge65756.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge65596.working, LeftRelationMerge65756.deltas,
    LeftRelationMerge65756.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 65756)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨22839⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22836⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22836⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge65596.working) (working := relationWorking0)
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
end SemanticResult65761

namespace SemanticResult65768
def owner : Owner := ⟨.program ⟨214⟩, ⟨30098⟩⟩
def rawTerms : List Term := Proof.Events256.exact65768RawTerms
def summary : Bound := (.finite 1292539135285018636288)
def resultEvent : Nat := 65768
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65768.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge65765.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult65761.owner)
    (rightOwner := SemanticResult65583.owner)
    (leftResult := 65761) (rightResult := 65583)
    (leftActual := SemanticResult65761.actual selector witness)
    (rightActual := SemanticResult65583.actual selector witness)
    (leftRaw := SemanticResult65761.rawTerms)
    (rightRaw := SemanticResult65583.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292539133473715126272) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 65762) (rightBinding := 65763)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22839⟩) (rightExpression := ⟨30097⟩)
    (coefficientTransfer := 65764) (summaryTransfer := 65767)
    (base := LeftOperatorMerge65765.base)
    (reconstruction := LeftOperatorMerge65765.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65761.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65583.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65765.operationAgreement
  · rfl
  · decide
end SemanticResult65768

namespace SemanticResult65775
def owner : Owner := ⟨.program ⟨214⟩, ⟨24726⟩⟩
def rawTerms : List Term := Proof.Events256.exact65775RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65775
def producerEvent : Nat := 65774
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65775.actual selector witness
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
end SemanticResult65775

namespace SemanticResult65778
def owner : Owner := ⟨.program ⟨214⟩, ⟨29806⟩⟩
def rawTerms : List Term := Proof.Events256.exact65778RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65778
def producerEvent : Nat := 65777
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65778.actual selector witness
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
end SemanticResult65778

namespace SemanticResult65785
def owner : Owner := ⟨.program ⟨214⟩, ⟨23372⟩⟩
def rawTerms : List Term := Proof.Events256.exact65785RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65785
def producerEvent : Nat := 65784
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65785.actual selector witness
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
end SemanticResult65785

namespace SemanticResult65788
def owner : Owner := ⟨.program ⟨214⟩, ⟨25676⟩⟩
def rawTerms : List Term := Proof.Events256.exact65788RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65788
def producerEvent : Nat := 65787
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65788.actual selector witness
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
end SemanticResult65788

namespace SemanticResult65793
def owner : Owner := ⟨.program ⟨214⟩, ⟨13149⟩⟩
def rawTerms : List Term := Proof.Events257.exact65793RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65793
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65793.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65792.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge65792.frameStart)
    (transferEvent := 65791) (owner := owner)
    (leftResult := 3109) (rightResult := 65295)
    (working := LeftOperatorMerge65792.working)
    (reconstruction := LeftOperatorMerge65792.reconstruction)
    (leftReference := .predecessor 0 65789 .coefficient) (rightReference := .predecessor 1 65790 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3109.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65792.operationAgreement
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
end SemanticResult65793

namespace SemanticResult65798
def owner : Owner := ⟨.program ⟨214⟩, ⟨7207⟩⟩
def rawTerms : List Term := Proof.Events257.exact65798RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65798
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65798.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65797.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge65797.frameStart)
    (transferEvent := 65796) (owner := owner)
    (leftResult := 65165) (rightResult := 6973)
    (working := LeftOperatorMerge65797.working)
    (reconstruction := LeftOperatorMerge65797.reconstruction)
    (leftReference := .predecessor 0 65794 .coefficient) (rightReference := .predecessor 1 65795 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult65165.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6973.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65797.operationAgreement
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
end SemanticResult65798

namespace SemanticResult65802
def owner : Owner := ⟨.program ⟨214⟩, ⟨13150⟩⟩
def rawTerms : List Term := Proof.Events257.exact65802RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65802
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65802.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 65799) (rightBinding := 65800)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7207⟩) (rightExpression := ⟨13149⟩)
    (transferEvent := 65801)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65798.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65793.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult65802

namespace SemanticResult65808
def owner : Owner := ⟨.program ⟨214⟩, ⟨13151⟩⟩
def rawTerms : List Term := Proof.Events257.exact65808RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 65808
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65808.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 65805) (survivorTransfer := 65806)
    (survivorEvent := 65807) (resultEvent := resultEvent)
    (rightCoefficientProducer := 6964)
    (owner := owner) (leftOwner := SemanticResult65802.owner)
    (rightOwner := SemanticResult6965.owner)
    (leftResult := 65802) (rightResult := 6965)
    (leftBinding := 65803) (rightBinding := 65804)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13150⟩) (rightExpression := ⟨103⟩)
    (leftActual := SemanticResult65802.actual selector witness)
    (rightActual := SemanticResult6965.actual selector witness)
    (leftRaw := SemanticResult65802.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨103⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound6964.actual selector witness)
    (survivorMagnitude := LeftBound65806.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65802.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult6965.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6964.derived selector witness)
  · exact LeftBound65806.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult65808

namespace SemanticResult65816
def owner : Owner := ⟨.program ⟨214⟩, ⟨13152⟩⟩
def rawTerms : List Term := Proof.Events257.exact65816RawTerms
def summary : Bound := (.finite 48256)
def resultEvent : Nat := 65816
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65816.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨58, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65814.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge65814.frameStart)
    (owner := owner) (leftOwner := SemanticResult65808.owner)
    (rightOwner := SemanticResult3112.owner)
    (leftResult := 65808) (rightResult := 3112)
    (leftActual := SemanticResult65808.actual selector witness)
    (rightActual := SemanticResult3112.actual selector witness)
    (leftRaw := SemanticResult65808.rawTerms)
    (rightRaw := SemanticResult3112.rawTerms)
    (working := LeftOperatorMerge65814.working)
    (leftBinding := 65809) (rightBinding := 65810)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨13151⟩) (rightExpression := ⟨10235⟩)
    (coefficientTransfer := 65811) (summaryTransfer := 65813)
    (rightCoefficientProducer := 3111)
    (rightSummaryTransfer := 65812)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨58, by decide⟩)
    (rightRecordedMaximum := 58)
    (rightSummaryMaximum := ⟨58, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge65814.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority3111.actual selector witness)
    (summaryMagnitude := LeftBound65813.actual selector witness)
    (reconstruction := LeftOperatorMerge65814.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult65808.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult3112.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3111.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority3111.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge65814.operationAgreement
  · exact LeftBound65813.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65814.working summary) := by
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
end SemanticResult65816

namespace SemanticResult65821
def owner : Owner := ⟨.program ⟨214⟩, ⟨10236⟩⟩
def rawTerms : List Term := Proof.Events257.exact65821RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 65821
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult65821.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge65820.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge65820.frameStart)
    (transferEvent := 65819) (owner := owner)
    (leftResult := 3112) (rightResult := 65295)
    (working := LeftOperatorMerge65820.working)
    (reconstruction := LeftOperatorMerge65820.reconstruction)
    (leftReference := .predecessor 0 65817 .coefficient) (rightReference := .predecessor 1 65818 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult3112.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult65295.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge65820.operationAgreement
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
end SemanticResult65821

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
