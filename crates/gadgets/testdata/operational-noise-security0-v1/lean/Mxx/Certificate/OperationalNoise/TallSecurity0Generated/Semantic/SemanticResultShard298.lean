import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Cert.ResidualShard298
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard014
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard093
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard094
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard263
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard264
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticResultShard297

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult40399
def owner : Owner := ⟨.program ⟨214⟩, ⟨11562⟩⟩
def rawTerms : List Term := Proof.Events157.exact40399RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40399
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40399.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40398.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge40398.frameStart)
    (transferEvent := 40397) (owner := owner)
    (leftResult := 1797) (rightResult := 36045)
    (working := LeftOperatorMerge40398.working)
    (reconstruction := LeftOperatorMerge40398.reconstruction)
    (leftReference := .predecessor 0 40395 .coefficient) (rightReference := .predecessor 1 40396 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1797.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40398.operationAgreement
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
end SemanticResult40399

namespace SemanticResult40404
def owner : Owner := ⟨.program ⟨214⟩, ⟨7312⟩⟩
def rawTerms : List Term := Proof.Events157.exact40404RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40404
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40404.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40403.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge40403.frameStart)
    (transferEvent := 40402) (owner := owner)
    (leftResult := 35915) (rightResult := 10981)
    (working := LeftOperatorMerge40403.working)
    (reconstruction := LeftOperatorMerge40403.reconstruction)
    (leftReference := .predecessor 0 40400 .coefficient) (rightReference := .predecessor 1 40401 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10981.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40403.operationAgreement
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
end SemanticResult40404

namespace SemanticResult40408
def owner : Owner := ⟨.program ⟨214⟩, ⟨11563⟩⟩
def rawTerms : List Term := Proof.Events157.exact40408RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40408
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40408.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 40405) (rightBinding := 40406)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7312⟩) (rightExpression := ⟨11562⟩)
    (transferEvent := 40407)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40404.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40399.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40408

namespace SemanticResult40414
def owner : Owner := ⟨.program ⟨214⟩, ⟨11564⟩⟩
def rawTerms : List Term := Proof.Events157.exact40414RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 40414
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40414.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 40411) (survivorTransfer := 40412)
    (survivorEvent := 40413) (resultEvent := resultEvent)
    (rightCoefficientProducer := 10972)
    (owner := owner) (leftOwner := SemanticResult40408.owner)
    (rightOwner := SemanticResult10973.owner)
    (leftResult := 40408) (rightResult := 10973)
    (leftBinding := 40409) (rightBinding := 40410)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11563⟩) (rightExpression := ⟨94⟩)
    (leftActual := SemanticResult40408.actual selector witness)
    (rightActual := SemanticResult10973.actual selector witness)
    (leftRaw := SemanticResult40408.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound10972.actual selector witness)
    (survivorMagnitude := LeftBound40412.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40408.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult10973.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)
  · exact LeftBound40412.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult40414

namespace SemanticResult40422
def owner : Owner := ⟨.program ⟨214⟩, ⟨14445⟩⟩
def rawTerms : List Term := Proof.Events157.exact40422RawTerms
def summary : Bound := (.finite 18304)
def resultEvent : Nat := 40422
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40422.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 32
      (.finite ⟨26, by decide⟩)
      (.finite ⟨22, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40420.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge40420.frameStart)
    (owner := owner) (leftOwner := SemanticResult40414.owner)
    (rightOwner := SemanticResult1800.owner)
    (leftResult := 40414) (rightResult := 1800)
    (leftActual := SemanticResult40414.actual selector witness)
    (rightActual := SemanticResult1800.actual selector witness)
    (leftRaw := SemanticResult40414.rawTerms)
    (rightRaw := SemanticResult1800.rawTerms)
    (working := LeftOperatorMerge40420.working)
    (leftBinding := 40415) (rightBinding := 40416)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11564⟩) (rightExpression := ⟨14442⟩)
    (coefficientTransfer := 40417) (summaryTransfer := 40419)
    (rightCoefficientProducer := 1799)
    (rightSummaryTransfer := 40418)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨22, by decide⟩)
    (rightRecordedMaximum := 22)
    (rightSummaryMaximum := ⟨22, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 32)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge40420.base)
    (coefficientFacts := ⟨false, true, none, none, some 1⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority1799.actual selector witness)
    (summaryMagnitude := LeftBound40419.actual selector witness)
    (reconstruction := LeftOperatorMerge40420.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40414.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult1800.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1799.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority1799.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge40420.operationAgreement
  · exact LeftBound40419.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40420.working summary) := by
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
end SemanticResult40422

namespace SemanticResult40427
def owner : Owner := ⟨.program ⟨214⟩, ⟨14446⟩⟩
def rawTerms : List Term := Proof.Events157.exact40427RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40427
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40427.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40426.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge40426.frameStart)
    (transferEvent := 40425) (owner := owner)
    (leftResult := 1800) (rightResult := 36045)
    (working := LeftOperatorMerge40426.working)
    (reconstruction := LeftOperatorMerge40426.reconstruction)
    (leftReference := .predecessor 0 40423 .coefficient) (rightReference := .predecessor 1 40424 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult1800.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult36045.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40426.operationAgreement
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
end SemanticResult40427

namespace SemanticResult40432
def owner : Owner := ⟨.program ⟨214⟩, ⟨7293⟩⟩
def rawTerms : List Term := Proof.Events157.exact40432RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40432
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40432.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40431.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge40431.frameStart)
    (transferEvent := 40430) (owner := owner)
    (leftResult := 35915) (rightResult := 11022)
    (working := LeftOperatorMerge40431.working)
    (reconstruction := LeftOperatorMerge40431.reconstruction)
    (leftReference := .predecessor 0 40428 .coefficient) (rightReference := .predecessor 1 40429 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult35915.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11022.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40431.operationAgreement
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
end SemanticResult40432

namespace SemanticResult40436
def owner : Owner := ⟨.program ⟨214⟩, ⟨14447⟩⟩
def rawTerms : List Term := Proof.Events157.exact40436RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40436
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40436.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 40433) (rightBinding := 40434)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7293⟩) (rightExpression := ⟨14446⟩)
    (transferEvent := 40435)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40432.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40427.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult40436

namespace SemanticResult40442
def owner : Owner := ⟨.program ⟨214⟩, ⟨14448⟩⟩
def rawTerms : List Term := Proof.Events157.exact40442RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 40442
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40442.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (coefficientTransfer := 40439) (survivorTransfer := 40440)
    (survivorEvent := 40441) (resultEvent := resultEvent)
    (rightCoefficientProducer := 11013)
    (owner := owner) (leftOwner := SemanticResult40436.owner)
    (rightOwner := SemanticResult11014.owner)
    (leftResult := 40436) (rightResult := 11014)
    (leftBinding := 40437) (rightBinding := 40438)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14447⟩) (rightExpression := ⟨75⟩)
    (leftActual := SemanticResult40436.actual selector witness)
    (rightActual := SemanticResult11014.actual selector witness)
    (leftRaw := SemanticResult40436.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨214⟩, ⟨75⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound11013.actual selector witness)
    (survivorMagnitude := LeftBound40440.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40436.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11014.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)
  · exact LeftBound40440.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult40442

namespace SemanticResult40452
def owner : Owner := ⟨.program ⟨214⟩, ⟨14449⟩⟩
def rawTerms : List Term := Proof.Events158.exact40452RawTerms
def summary : Bound := (.finite 95420416)
def resultEvent : Nat := 40452
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40452.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨26, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40448.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge40448.frameStart)
    (owner := owner) (leftOwner := SemanticResult40442.owner)
    (rightOwner := SemanticResult11011.owner)
    (leftResult := 40442) (rightResult := 11011)
    (leftActual := SemanticResult40442.actual selector witness)
    (rightActual := SemanticResult11011.actual selector witness)
    (leftRaw := SemanticResult40442.rawTerms)
    (rightRaw := SemanticResult11011.rawTerms)
    (working := LeftOperatorMerge40448.working)
    (leftBinding := 40443) (rightBinding := 40444)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14448⟩) (rightExpression := ⟨7856⟩)
    (coefficientTransfer := 40445) (summaryTransfer := 40447)
    (rightCoefficientProducer := 11010)
    (rightSummaryTransfer := 40446)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge40448.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound11010.actual selector witness)
    (summaryMagnitude := LeftBound40447.actual selector witness)
    (reconstruction := LeftOperatorMerge40448.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40442.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult11011.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11010.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound11010.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge40448.operationAgreement
  · exact LeftBound40447.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40448.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 40449 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge40448.working
    [{ coefficient := (-1), key := LeftRelationMerge40449.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge40449.frameStart
      LeftRelationMerge40449.owner (.relation 40449) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge40449.deltas
    rows := LeftRelationMerge40449.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge40448.working LeftRelationMerge40449.source
        (relationContext LeftRelationMerge40449.source
          LeftRelationMerge40449.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge40448.working, LeftRelationMerge40449.deltas,
    LeftRelationMerge40449.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 40449)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨14449⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge40448.working) (working := relationWorking0)
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
end SemanticResult40452

namespace SemanticResult40458
def owner : Owner := ⟨.program ⟨214⟩, ⟨14450⟩⟩
def rawTerms : List Term := Proof.Events158.exact40458RawTerms
def summary : Bound := (.finite 95438720)
def resultEvent : Nat := 40458
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40458.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (frameStart := LeftOperatorMerge40456.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult40452.owner)
    (rightOwner := SemanticResult40422.owner)
    (leftResult := 40452) (rightResult := 40422)
    (leftActual := SemanticResult40452.actual selector witness)
    (rightActual := SemanticResult40422.actual selector witness)
    (leftRaw := SemanticResult40452.rawTerms)
    (rightRaw := SemanticResult40422.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 95420416)
    (rightMaximum := 18304) (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14)
    (leftBinding := 40453) (rightBinding := 40454)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14449⟩) (rightExpression := ⟨14445⟩)
    (coefficientTransfer := 40455) (summaryTransfer := 40457)
    (base := LeftOperatorMerge40456.base)
    (reconstruction := LeftOperatorMerge40456.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40452.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40422.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge40456.operationAgreement
  · rfl
  · decide
end SemanticResult40458

namespace SemanticResult40468
def owner : Owner := ⟨.program ⟨214⟩, ⟨26154⟩⟩
def rawTerms : List Term := Proof.Events158.exact40468RawTerms
def summary : Bound := (.finite 350261629419520)
def resultEvent : Nat := 40468
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40468.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 448
      (.finite ⟨95438720, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40464.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 100418593683253592432016548326729029359133068138294319235841)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge40464.frameStart)
    (owner := owner) (leftOwner := SemanticResult40458.owner)
    (rightOwner := SemanticResult40394.owner)
    (leftResult := 40458) (rightResult := 40394)
    (leftActual := SemanticResult40458.actual selector witness)
    (rightActual := SemanticResult40394.actual selector witness)
    (leftRaw := SemanticResult40458.rawTerms)
    (rightRaw := SemanticResult40394.rawTerms)
    (working := LeftOperatorMerge40464.working)
    (leftBinding := 40459) (rightBinding := 40460)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨14450⟩) (rightExpression := ⟨26153⟩)
    (coefficientTransfer := 40461) (summaryTransfer := 40463)
    (rightCoefficientProducer := 40393)
    (rightSummaryTransfer := 40462)
    (leftMaximum := ⟨95438720, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 448)
    (valueType := .matrix "100418593683253592432016548326729029359133068138294319235841" 32 1 14) (base := LeftOperatorMerge40464.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority40393.actual selector witness)
    (summaryMagnitude := LeftBound40463.actual selector witness)
    (reconstruction := LeftOperatorMerge40464.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult40458.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult40394.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority40393.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority40393.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge40464.operationAgreement
  · exact LeftBound40463.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact LeftOperatorMerge40464.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 40465 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6761⟩⟩, ⟨.program ⟨214⟩, ⟨7855⟩⟩, ⟨.program ⟨214⟩, ⟨26153⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11561⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23630⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23630⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge40464.working
    [{ coefficient := (-1), key := LeftRelationMerge40465.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge40465.frameStart
      LeftRelationMerge40465.owner (.relation 40465) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge40465.deltas
    rows := LeftRelationMerge40465.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge40464.working LeftRelationMerge40465.source
        (relationContext LeftRelationMerge40465.source
          LeftRelationMerge40465.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge40464.working, LeftRelationMerge40465.deltas,
    LeftRelationMerge40465.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ValueClaim.Interprets 100418593683253592432016548326729029359133068138294319235841 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 100418593683253592432016548326729029359133068138294319235841) (witness := witness) (application := 40465)
    (frameStart := 0) (owner := ⟨.program ⟨214⟩, ⟨26154⟩⟩)
    (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11561⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26153⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26153⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge40464.working) (working := relationWorking0)
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
end SemanticResult40468

namespace SemanticResult40471
def owner : Owner := ⟨.program ⟨214⟩, ⟨19608⟩⟩
def rawTerms : List Term := Proof.Events158.exact40471RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40471
def producerEvent : Nat := 40470
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40471.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨16⟩), 0, .finite 136065468, .authorityRelationPreimageSource ⟨16⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult40471

namespace SemanticResult40475
def owner : Owner := ⟨.program ⟨214⟩, ⟨19610⟩⟩
def rawTerms : List Term := Proof.Events158.exact40475RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40475
def producerEvent : Nat := 40474
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40475.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 40472 .coefficient) (.value (.predecessor 1 40473 .coefficient)), 0, .finite 136065468, .scale (.predecessor 0 40472 .coefficient) (.value (.predecessor 1 40473 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult40475

namespace SemanticResult40553
def owner : Owner := ⟨.program ⟨214⟩, ⟨11561⟩⟩
def rawTerms : List Term := Proof.Events158.exact40553RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40553
def producerEvent : Nat := 40552
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40553.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 40530, .finite 22, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult40553

namespace SemanticResult40556
def owner : Owner := ⟨.program ⟨214⟩, ⟨14442⟩⟩
def rawTerms : List Term := Proof.Events158.exact40556RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 40556
def producerEvent : Nat := 40555
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Int :=
  Cert.ResidualResult40556.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 40530, .finite 22, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    ExactClaimAt history 100418593683253592432016548326729029359133068138294319235841 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult40556

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
