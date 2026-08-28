import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard368
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard367

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult51086
def owner : Owner := ⟨.program ⟨214⟩, ⟨17056⟩⟩
def rawTerms : List Term := Proof.Events199.exact51086RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51086
def producerEvent : Nat := 51085
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51086.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.identity (.predecessor 0 51084 .coefficient), 51026, .finite 60, .identity (.predecessor 0 51084 .coefficient), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51086

namespace SemanticResult51088
def owner : Owner := ⟨.program ⟨214⟩, ⟨6544⟩⟩
def rawTerms : List Term := Proof.Events199.exact51088RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51088
def producerEvent : Nat := 51087
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51088.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.factStore), 51026, .large, .authorityFactStore, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51088

namespace SemanticResult51093
def owner : Owner := ⟨.program ⟨214⟩, ⟨17057⟩⟩
def rawTerms : List Term := Proof.Events199.exact51093RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51093
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51093.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51092.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge51092.frameStart)
    (transferEvent := 51091) (owner := owner)
    (leftResult := 51088) (rightResult := 51086)
    (working := LeftOperatorMerge51092.working)
    (reconstruction := LeftOperatorMerge51092.reconstruction)
    (leftReference := .predecessor 0 51089 .coefficient) (rightReference := .predecessor 1 51090 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult51088.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51086.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge51092.operationAgreement
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
end SemanticResult51093

namespace SemanticResult51096
def owner : Owner := ⟨.program ⟨214⟩, ⟨6707⟩⟩
def rawTerms : List Term := Proof.Events199.exact51096RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51096
def producerEvent : Nat := 51095
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51096.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 51026, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51096

namespace SemanticResult51100
def owner : Owner := ⟨.program ⟨214⟩, ⟨17058⟩⟩
def rawTerms : List Term := Proof.Events199.exact51100RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51100
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51100.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 51097) (rightBinding := 51098)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6707⟩) (rightExpression := ⟨17057⟩)
    (transferEvent := 51099)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51096.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51093.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult51100

namespace SemanticResult51108
def owner : Owner := ⟨.program ⟨214⟩, ⟨30140⟩⟩
def rawTerms : List Term := Proof.Events199.exact51108RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51108
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51108.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51104.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge51104.frameStart)
    (transferEvent := 51103) (owner := owner)
    (leftResult := 51100) (rightResult := 51077)
    (working := LeftOperatorMerge51104.working)
    (reconstruction := LeftOperatorMerge51104.reconstruction)
    (leftReference := .predecessor 0 51101 .coefficient) (rightReference := .predecessor 1 51102 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult51100.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51077.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge51104.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 51106 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24795⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24795⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge51104.working
    [{ coefficient := (-1), key := LeftRelationMerge51106.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge51106.frameStart
      LeftRelationMerge51106.owner (.relation 51106) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge51106.deltas
    rows := LeftRelationMerge51106.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge51104.working LeftRelationMerge51106.source
        (relationContext LeftRelationMerge51106.source
          LeftRelationMerge51106.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge51104.working, LeftRelationMerge51106.deltas,
    LeftRelationMerge51106.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 51106)
    (frameStart := 51026) (owner := ⟨.program ⟨214⟩, ⟨30140⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨17015⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge51104.working) (working := relationWorking0)
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
end SemanticResult51108

namespace SemanticResult51111
def owner : Owner := ⟨.program ⟨214⟩, ⟨18173⟩⟩
def rawTerms : List Term := Proof.Events199.exact51111RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51111
def producerEvent : Nat := 51110
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51111.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 51026, .finite 63, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51111

namespace SemanticResult51116
def owner : Owner := ⟨.program ⟨214⟩, ⟨18174⟩⟩
def rawTerms : List Term := Proof.Events199.exact51116RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51116
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51116.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge51115.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge51115.frameStart)
    (transferEvent := 51114) (owner := owner)
    (leftResult := 51088) (rightResult := 51111)
    (working := LeftOperatorMerge51115.working)
    (reconstruction := LeftOperatorMerge51115.reconstruction)
    (leftReference := .predecessor 0 51112 .coefficient) (rightReference := .predecessor 1 51113 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult51088.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51111.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge51115.operationAgreement
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
end SemanticResult51116

namespace SemanticResult51119
def owner : Owner := ⟨.program ⟨214⟩, ⟨6743⟩⟩
def rawTerms : List Term := Proof.Events199.exact51119RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51119
def producerEvent : Nat := 51118
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51119.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 51026, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult51119

namespace SemanticResult51123
def owner : Owner := ⟨.program ⟨214⟩, ⟨18175⟩⟩
def rawTerms : List Term := Proof.Events199.exact51123RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51123
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51123.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 51120) (rightBinding := 51121)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨6743⟩) (rightExpression := ⟨18174⟩)
    (transferEvent := 51122)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51119.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51116.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult51123

namespace SemanticResult51127
def owner : Owner := ⟨.program ⟨214⟩, ⟨30147⟩⟩
def rawTerms : List Term := Proof.Events199.exact51127RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51127
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51127.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 51124) (rightBinding := 51125)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨18175⟩) (rightExpression := ⟨30140⟩)
    (transferEvent := 51126)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51123.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult51108.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult51127

namespace SemanticResult51136
def owner : Owner := ⟨.program ⟨214⟩, ⟨22847⟩⟩
def rawTerms : List Term := Proof.Events199.exact51136RawTerms
def summary : Bound := (.finite 1811303510016)
def resultEvent : Nat := 51136
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51136.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 512
      (.finite ⟨26, by decide⟩)
      (.finite ⟨136065468, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50971.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge50971.frameStart)
    (owner := owner) (leftOwner := SemanticResult50762.owner)
    (rightOwner := SemanticResult50965.owner)
    (leftResult := 50762) (rightResult := 50965)
    (leftActual := SemanticResult50762.actual selector witness)
    (rightActual := SemanticResult50965.actual selector witness)
    (leftRaw := SemanticResult50762.rawTerms)
    (rightRaw := SemanticResult50965.rawTerms)
    (working := LeftOperatorMerge50971.working)
    (leftBinding := 50966) (rightBinding := 50967)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5547⟩) (rightExpression := ⟨22846⟩)
    (coefficientTransfer := 50968) (summaryTransfer := 50970)
    (rightCoefficientProducer := 50964)
    (rightSummaryTransfer := 50969)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨136065468, by decide⟩)
    (rightRecordedMaximum := 136065468)
    (rightSummaryMaximum := ⟨136065468, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 512)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge50971.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound50964.actual selector witness)
    (summaryMagnitude := LeftBound50970.actual selector witness)
    (reconstruction := LeftOperatorMerge50971.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult50762.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50965.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50964.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound50964.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge50971.operationAgreement
  · exact LeftBound50970.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge50971.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 51131 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24795⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨18173⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30139⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17015⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24795⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18173⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge50971.working
    [{ coefficient := (1), key := LeftRelationMerge51131.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge51131.frameStart
      LeftRelationMerge51131.owner (.relation 51131) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge51131.deltas
    rows := LeftRelationMerge51131.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge50971.working LeftRelationMerge51131.source
        (relationContext LeftRelationMerge51131.source
          LeftRelationMerge51131.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge50971.working, LeftRelationMerge51131.deltas,
    LeftRelationMerge51131.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 51131)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨22847⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22844⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22844⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge50971.working) (working := relationWorking0)
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
end SemanticResult51136

namespace SemanticResult51143
def owner : Owner := ⟨.program ⟨214⟩, ⟨30142⟩⟩
def rawTerms : List Term := Proof.Events199.exact51143RawTerms
def summary : Bound := (.finite 1292539135285018636288)
def resultEvent : Nat := 51143
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51143.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge51140.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult51136.owner)
    (rightOwner := SemanticResult50958.owner)
    (leftResult := 51136) (rightResult := 50958)
    (leftActual := SemanticResult51136.actual selector witness)
    (rightActual := SemanticResult50958.actual selector witness)
    (leftRaw := SemanticResult51136.rawTerms)
    (rightRaw := SemanticResult50958.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1811303510016)
    (rightMaximum := 1292539133473715126272) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 51137) (rightBinding := 51138)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨22847⟩) (rightExpression := ⟨30141⟩)
    (coefficientTransfer := 51139) (summaryTransfer := 51142)
    (base := LeftOperatorMerge51140.base)
    (reconstruction := LeftOperatorMerge51140.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult51136.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult50958.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge51140.operationAgreement
  · rfl
  · decide
end SemanticResult51143

namespace SemanticResult51150
def owner : Owner := ⟨.program ⟨214⟩, ⟨24732⟩⟩
def rawTerms : List Term := Proof.Events199.exact51150RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51150
def producerEvent : Nat := 51149
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51150.actual selector witness
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
end SemanticResult51150

namespace SemanticResult51153
def owner : Owner := ⟨.program ⟨214⟩, ⟨29832⟩⟩
def rawTerms : List Term := Proof.Events199.exact51153RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51153
def producerEvent : Nat := 51152
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51153.actual selector witness
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
end SemanticResult51153

namespace SemanticResult51160
def owner : Owner := ⟨.program ⟨214⟩, ⟨23376⟩⟩
def rawTerms : List Term := Proof.Events199.exact51160RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 51160
def producerEvent : Nat := 51159
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult51160.actual selector witness
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
end SemanticResult51160

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
