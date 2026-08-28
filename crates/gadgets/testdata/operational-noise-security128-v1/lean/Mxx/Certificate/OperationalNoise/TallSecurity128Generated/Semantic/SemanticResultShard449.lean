import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard449
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard006
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard127
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard128
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard350
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard351
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard352
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard439
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard440
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard442
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard443
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard445
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard446
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard447
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard448

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult60749
def owner : Owner := ⟨.program ⟨257⟩, ⟨16755⟩⟩
def rawTerms : List Term := Proof.Events237.exact60749RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 60749
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60749.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge60584.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge60584.frameStart)
    (owner := owner) (leftOwner := SemanticResult46745.owner)
    (rightOwner := SemanticResult60578.owner)
    (leftResult := 46745) (rightResult := 60578)
    (leftActual := SemanticResult46745.actual selector witness)
    (rightActual := SemanticResult60578.actual selector witness)
    (leftRaw := SemanticResult46745.rawTerms)
    (rightRaw := SemanticResult60578.rawTerms)
    (working := LeftOperatorMerge60584.working)
    (leftBinding := 60579) (rightBinding := 60580)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11216⟩) (rightExpression := ⟨16754⟩)
    (coefficientTransfer := 60581) (summaryTransfer := 60583)
    (rightCoefficientProducer := 60577)
    (rightSummaryTransfer := 60582)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge60584.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound60577.actual selector witness)
    (summaryMagnitude := LeftBound60583.actual selector witness)
    (reconstruction := LeftOperatorMerge60584.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult46745.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60578.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60577.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound60577.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge60584.operationAgreement
  · exact LeftBound60583.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge60584.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 60744 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨15852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17072⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17978⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17072⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge60584.working
    [{ coefficient := (1), key := LeftRelationMerge60744.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge60744.frameStart
      LeftRelationMerge60744.owner (.relation 60744) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge60744.deltas
    rows := LeftRelationMerge60744.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge60584.working LeftRelationMerge60744.source
        (relationContext LeftRelationMerge60744.source
          LeftRelationMerge60744.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge60584.working, LeftRelationMerge60744.deltas,
    LeftRelationMerge60744.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 60744)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16755⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16752⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16752⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge60584.working) (working := relationWorking0)
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
end SemanticResult60749

namespace SemanticResult60756
def owner : Owner := ⟨.program ⟨257⟩, ⟨17981⟩⟩
def rawTerms : List Term := Proof.Events237.exact60756RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 60756
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60756.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge60753.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60749.owner)
    (rightOwner := SemanticResult60571.owner)
    (leftResult := 60749) (rightResult := 60571)
    (leftActual := SemanticResult60749.actual selector witness)
    (rightActual := SemanticResult60571.actual selector witness)
    (leftRaw := SemanticResult60749.rawTerms)
    (rightRaw := SemanticResult60571.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60750) (rightBinding := 60751)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16755⟩) (rightExpression := ⟨17980⟩)
    (coefficientTransfer := 60752) (summaryTransfer := 60755)
    (base := LeftOperatorMerge60753.base)
    (reconstruction := LeftOperatorMerge60753.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60749.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60571.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge60753.operationAgreement
  · rfl
  · decide
end SemanticResult60756

namespace SemanticResult60766
def owner : Owner := ⟨.program ⟨257⟩, ⟨17982⟩⟩
def rawTerms : List Term := Proof.Events237.exact60766RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529920)
def resultEvent : Nat := 60766
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60766.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨32188807212483706889510625476608, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge60762.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge60762.frameStart)
    (owner := owner) (leftOwner := SemanticResult60756.owner)
    (rightOwner := SemanticResult15882.owner)
    (leftResult := 60756) (rightResult := 15882)
    (leftActual := SemanticResult60756.actual selector witness)
    (rightActual := SemanticResult15882.actual selector witness)
    (leftRaw := SemanticResult60756.rawTerms)
    (rightRaw := SemanticResult15882.rawTerms)
    (working := LeftOperatorMerge60762.working)
    (leftBinding := 60757) (rightBinding := 60758)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17981⟩) (rightExpression := ⟨7172⟩)
    (coefficientTransfer := 60759) (summaryTransfer := 60761)
    (rightCoefficientProducer := 15881)
    (rightSummaryTransfer := 60760)
    (leftMaximum := ⟨32188807212483706889510625476608, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge60762.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15881.actual selector witness)
    (summaryMagnitude := LeftBound60761.actual selector witness)
    (reconstruction := LeftOperatorMerge60762.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60756.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15882.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15881.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15881.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge60762.operationAgreement
  · exact LeftBound60761.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge60762.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 60764 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7197⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge60762.working
    [{ coefficient := (-1), key := LeftRelationMerge60764.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge60764.frameStart
      LeftRelationMerge60764.owner (.relation 60764) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge60764.deltas
    rows := LeftRelationMerge60764.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge60762.working LeftRelationMerge60764.source
        (relationContext LeftRelationMerge60764.source
          LeftRelationMerge60764.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge60762.working, LeftRelationMerge60764.deltas,
    LeftRelationMerge60764.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 60764)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨17982⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨16158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge60762.working) (working := relationWorking0)
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
end SemanticResult60766

namespace SemanticResult60771
def owner : Owner := ⟨.program ⟨257⟩, ⟨11220⟩⟩
def rawTerms : List Term := Proof.Events237.exact60771RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60771
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60771.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge60770.working .exactZero) := by
  apply operatorTensorMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge60770.frameStart)
    (transferEvent := 60769) (owner := owner)
    (leftResult := 723) (rightResult := 46653)
    (working := LeftOperatorMerge60770.working)
    (reconstruction := LeftOperatorMerge60770.reconstruction)
    (leftReference := .predecessor 0 60767 .coefficient) (rightReference := .predecessor 1 60768 .coefficient)
  · rfl
  · rfl
  · exact SemanticResult723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult46653.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge60770.operationAgreement
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
end SemanticResult60771

namespace SemanticResult60776
def owner : Owner := ⟨.program ⟨257⟩, ⟨11198⟩⟩
def rawTerms : List Term := Proof.Events237.exact60776RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60776
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60776.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge60775.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge60775.frameStart)
    (transferEvent := 60774) (owner := owner)
    (leftResult := 46523) (rightResult := 15896)
    (working := LeftOperatorMerge60775.working)
    (reconstruction := LeftOperatorMerge60775.reconstruction)
    (leftReference := .predecessor 0 60772 .coefficient) (rightReference := .predecessor 1 60773 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := true) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult46523.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15896.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge60775.operationAgreement
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
end SemanticResult60776

namespace SemanticResult60780
def owner : Owner := ⟨.program ⟨257⟩, ⟨11221⟩⟩
def rawTerms : List Term := Proof.Events237.exact60780RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 60780
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60780.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 60777) (rightBinding := 60778)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11198⟩) (rightExpression := ⟨11220⟩)
    (transferEvent := 60779)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60776.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60771.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60780

namespace SemanticResult60786
def owner : Owner := ⟨.program ⟨257⟩, ⟨11222⟩⟩
def rawTerms : List Term := Proof.Events237.exact60786RawTerms
def summary : Bound := (.finite 26)
def resultEvent : Nat := 60786
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60786.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddSingletonSurvivorFoldClaimAt
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (witness := witness) (frameStart := 0)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (coefficientTransfer := 60783) (survivorTransfer := 60784)
    (survivorEvent := 60785) (resultEvent := resultEvent)
    (rightCoefficientProducer := 31515)
    (owner := owner) (leftOwner := SemanticResult60780.owner)
    (rightOwner := SemanticResult31516.owner)
    (leftResult := 60780) (rightResult := 31516)
    (leftBinding := 60781) (rightBinding := 60782)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11221⟩) (rightExpression := ⟨118⟩)
    (leftActual := SemanticResult60780.actual selector witness)
    (rightActual := SemanticResult31516.actual selector witness)
    (leftRaw := SemanticResult60780.rawTerms)
    (survivorMonomial := ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩) (maximum := ⟨26, by decide⟩)
    (rightMagnitude := LeftBound31515.actual selector witness)
    (survivorMagnitude := LeftBound60784.actual selector witness)
  · decide +kernel
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60780.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult31516.claimSound selector selectorLower selectorUpper witness
  · rfl
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)
  · exact LeftBound60784.derived selector witness
  · rfl
  · rfl
  · decide
end SemanticResult60786

namespace SemanticResult60793
def owner : Owner := ⟨.program ⟨257⟩, ⟨11223⟩⟩
def rawTerms : List Term := Proof.Events237.exact60793RawTerms
def summary : Bound := (.finite 52)
def resultEvent : Nat := 60793
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60793.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge60790.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60786.owner)
    (rightOwner := SemanticResult60786.owner)
    (leftResult := 60786) (rightResult := 60786)
    (leftActual := SemanticResult60786.actual selector witness)
    (rightActual := SemanticResult60786.actual selector witness)
    (leftRaw := SemanticResult60786.rawTerms)
    (rightRaw := SemanticResult60786.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 26)
    (rightMaximum := 26) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60787) (rightBinding := 60788)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11222⟩) (rightExpression := ⟨11222⟩)
    (coefficientTransfer := 60789) (summaryTransfer := 60792)
    (base := LeftOperatorMerge60790.base)
    (reconstruction := LeftOperatorMerge60790.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60786.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60786.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge60790.operationAgreement
  · rfl
  · decide
end SemanticResult60793

namespace SemanticResult60798
def owner : Owner := ⟨.program ⟨257⟩, ⟨17983⟩⟩
def rawTerms : List Term := Proof.Events237.exact60798RawTerms
def summary : Bound := (.finite 345624685687166110058245054666339432529972)
def resultEvent : Nat := 60798
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60798.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60793.owner)
    (rightOwner := SemanticResult60766.owner)
    (leftResult := 60793) (rightResult := 60766)
    (leftActual := SemanticResult60793.actual selector witness)
    (rightActual := SemanticResult60766.actual selector witness)
    (leftRaw := SemanticResult60793.rawTerms)
    (rightRaw := SemanticResult60766.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 52)
    (rightMaximum := 345624685687166110058245054666339432529920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60794) (rightBinding := 60795)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨11223⟩) (rightExpression := ⟨17982⟩)
    (transferEvent := 60796) (summaryTransferEvent := 60797)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60793.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60766.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60798

namespace SemanticResult60803
def owner : Owner := ⟨.program ⟨257⟩, ⟨20898⟩⟩
def rawTerms : List Term := Proof.Events237.exact60803RawTerms
def summary : Bound := (.finite 691250426059631610003352154589745737891892)
def resultEvent : Nat := 60803
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60803.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60798.owner)
    (rightOwner := SemanticResult60554.owner)
    (leftResult := 60798) (rightResult := 60554)
    (leftActual := SemanticResult60798.actual selector witness)
    (rightActual := SemanticResult60554.actual selector witness)
    (leftRaw := SemanticResult60798.rawTerms)
    (rightRaw := SemanticResult60554.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 345624685687166110058245054666339432529972)
    (rightMaximum := 345625740372465499945107099923406305361920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60799) (rightBinding := 60800)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17983⟩) (rightExpression := ⟨20897⟩)
    (transferEvent := 60801) (summaryTransferEvent := 60802)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60798.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60554.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60803

namespace SemanticResult60808
def owner : Owner := ⟨.program ⟨257⟩, ⟨24118⟩⟩
def rawTerms : List Term := Proof.Events237.exact60808RawTerms
def summary : Bound := (.finite 1036877221117396499835321299770218916085812)
def resultEvent : Nat := 60808
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60808.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60803.owner)
    (rightOwner := SemanticResult60342.owner)
    (leftResult := 60803) (rightResult := 60342)
    (leftActual := SemanticResult60803.actual selector witness)
    (rightActual := SemanticResult60342.actual selector witness)
    (leftRaw := SemanticResult60803.rawTerms)
    (rightRaw := SemanticResult60342.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 691250426059631610003352154589745737891892)
    (rightMaximum := 345626795057764889831969145180473178193920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60804) (rightBinding := 60805)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20898⟩) (rightExpression := ⟨24117⟩)
    (transferEvent := 60806) (summaryTransferEvent := 60807)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60803.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60342.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60808

namespace SemanticResult60813
def owner : Owner := ⟨.program ⟨257⟩, ⟨34138⟩⟩
def rawTerms : List Term := Proof.Events237.exact60813RawTerms
def summary : Bound := (.finite 1382506125545760169441014535464825839943732)
def resultEvent : Nat := 60813
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60813.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60808.owner)
    (rightOwner := SemanticResult60130.owner)
    (leftResult := 60808) (rightResult := 60130)
    (leftActual := SemanticResult60808.actual selector witness)
    (rightActual := SemanticResult60130.actual selector witness)
    (leftRaw := SemanticResult60808.rawTerms)
    (rightRaw := SemanticResult60130.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1036877221117396499835321299770218916085812)
    (rightMaximum := 345628904428363669605693235694606923857920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60809) (rightBinding := 60810)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24118⟩) (rightExpression := ⟨34137⟩)
    (transferEvent := 60811) (summaryTransferEvent := 60812)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60808.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult60130.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60813

namespace SemanticResult60818
def owner : Owner := ⟨.program ⟨257⟩, ⟨53198⟩⟩
def rawTerms : List Term := Proof.Events237.exact60818RawTerms
def summary : Bound := (.finite 1728139248715321398594155952187700255129652)
def resultEvent : Nat := 60818
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60818.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60813.owner)
    (rightOwner := SemanticResult59918.owner)
    (leftResult := 60813) (rightResult := 59918)
    (leftActual := SemanticResult60813.actual selector witness)
    (rightActual := SemanticResult59918.actual selector witness)
    (leftRaw := SemanticResult60813.rawTerms)
    (rightRaw := SemanticResult59918.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1382506125545760169441014535464825839943732)
    (rightMaximum := 345633123169561229153141416722874415185920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60814) (rightBinding := 60815)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34138⟩) (rightExpression := ⟨53197⟩)
    (transferEvent := 60816) (summaryTransferEvent := 60817)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60813.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59918.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60818

namespace SemanticResult60823
def owner : Owner := ⟨.program ⟨257⟩, ⟨56178⟩⟩
def rawTerms : List Term := Proof.Events237.exact60823RawTerms
def summary : Bound := (.finite 2073774481255481407521021459424708415979572)
def resultEvent : Nat := 60823
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60823.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60818.owner)
    (rightOwner := SemanticResult59706.owner)
    (leftResult := 60818) (rightResult := 59706)
    (leftActual := SemanticResult60818.actual selector witness)
    (rightActual := SemanticResult59706.actual selector witness)
    (leftRaw := SemanticResult60818.rawTerms)
    (rightRaw := SemanticResult59706.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 1728139248715321398594155952187700255129652)
    (rightMaximum := 345635232540160008926865507237008160849920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60819) (rightBinding := 60820)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53198⟩) (rightExpression := ⟨56177⟩)
    (transferEvent := 60821) (summaryTransferEvent := 60822)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60818.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59706.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60823

namespace SemanticResult60828
def owner : Owner := ⟨.program ⟨257⟩, ⟨59158⟩⟩
def rawTerms : List Term := Proof.Events237.exact60828RawTerms
def summary : Bound := (.finite 2419413932536838975995335147689984068157492)
def resultEvent : Nat := 60828
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60828.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60823.owner)
    (rightOwner := SemanticResult59494.owner)
    (leftResult := 60823) (rightResult := 59494)
    (leftActual := SemanticResult60823.actual selector witness)
    (rightActual := SemanticResult59494.actual selector witness)
    (leftRaw := SemanticResult60823.rawTerms)
    (rightRaw := SemanticResult59494.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2073774481255481407521021459424708415979572)
    (rightMaximum := 345639451281357568474313688265275652177920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60824) (rightBinding := 60825)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56178⟩) (rightExpression := ⟨59157⟩)
    (transferEvent := 60826) (summaryTransferEvent := 60827)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60823.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59494.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60828

namespace SemanticResult60833
def owner : Owner := ⟨.program ⟨257⟩, ⟨62138⟩⟩
def rawTerms : List Term := Proof.Events237.exact60833RawTerms
def summary : Bound := (.finite 2765055493188795324243372926469393465999412)
def resultEvent : Nat := 60833
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult60833.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult60828.owner)
    (rightOwner := SemanticResult59282.owner)
    (leftResult := 60828) (rightResult := 59282)
    (leftActual := SemanticResult60828.actual selector witness)
    (rightActual := SemanticResult59282.actual selector witness)
    (leftRaw := SemanticResult60828.rawTerms)
    (rightRaw := SemanticResult59282.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 2419413932536838975995335147689984068157492)
    (rightMaximum := 345641560651956348248037778779409397841920) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 60829) (rightBinding := 60830)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59158⟩) (rightExpression := ⟨62137⟩)
    (transferEvent := 60831) (summaryTransferEvent := 60832)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult60828.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult59282.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult60833

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
