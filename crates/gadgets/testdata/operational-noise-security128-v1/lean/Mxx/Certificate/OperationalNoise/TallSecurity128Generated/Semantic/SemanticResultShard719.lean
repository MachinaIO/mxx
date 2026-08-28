import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard719
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard654
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard693
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard697
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard701
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard704
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard708
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard712
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard715
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard718

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult99152
def owner : Owner := ⟨.program ⟨257⟩, ⟨17229⟩⟩
def rawTerms : List Term := Proof.Events387.exact99152RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99152
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99152.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 99149) (rightBinding := 99150)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7179⟩) (rightExpression := ⟨17228⟩)
    (transferEvent := 99151)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99148.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99145.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99152

namespace SemanticResult99160
def owner : Owner := ⟨.program ⟨257⟩, ⟨17902⟩⟩
def rawTerms : List Term := Proof.Events387.exact99160RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99160
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99160.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99156.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99156.frameStart)
    (transferEvent := 99155) (owner := owner)
    (leftResult := 99152) (rightResult := 99129)
    (working := LeftOperatorMerge99156.working)
    (reconstruction := LeftOperatorMerge99156.reconstruction)
    (leftReference := .predecessor 0 99153 .coefficient) (rightReference := .predecessor 1 99154 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult99152.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99129.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99156.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 99158 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge99156.working
    [{ coefficient := (-1), key := LeftRelationMerge99158.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge99158.frameStart
      LeftRelationMerge99158.owner (.relation 99158) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge99158.deltas
    rows := LeftRelationMerge99158.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge99156.working LeftRelationMerge99158.source
        (relationContext LeftRelationMerge99158.source
          LeftRelationMerge99158.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge99156.working, LeftRelationMerge99158.deltas,
    LeftRelationMerge99158.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 99158)
    (frameStart := 99078) (owner := ⟨.program ⟨257⟩, ⟨17902⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge99156.working) (working := relationWorking0)
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
end SemanticResult99160

namespace SemanticResult99163
def owner : Owner := ⟨.program ⟨257⟩, ⟨16115⟩⟩
def rawTerms : List Term := Proof.Events387.exact99163RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99163
def producerEvent : Nat := 99162
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99163.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 99078, .finite 43, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult99163

namespace SemanticResult99168
def owner : Owner := ⟨.program ⟨257⟩, ⟨16116⟩⟩
def rawTerms : List Term := Proof.Events387.exact99168RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99168
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99168.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99167.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge99167.frameStart)
    (transferEvent := 99166) (owner := owner)
    (leftResult := 99140) (rightResult := 99163)
    (working := LeftOperatorMerge99167.working)
    (reconstruction := LeftOperatorMerge99167.reconstruction)
    (leftReference := .predecessor 0 99164 .coefficient) (rightReference := .predecessor 1 99165 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult99140.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99163.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99167.operationAgreement
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
end SemanticResult99168

namespace SemanticResult99171
def owner : Owner := ⟨.program ⟨257⟩, ⟨7198⟩⟩
def rawTerms : List Term := Proof.Events387.exact99171RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99171
def producerEvent : Nat := 99170
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99171.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 99078, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult99171

namespace SemanticResult99175
def owner : Owner := ⟨.program ⟨257⟩, ⟨16117⟩⟩
def rawTerms : List Term := Proof.Events387.exact99175RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99175
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99175.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 99172) (rightBinding := 99173)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7198⟩) (rightExpression := ⟨16116⟩)
    (transferEvent := 99174)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99171.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99168.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99175

namespace SemanticResult99179
def owner : Owner := ⟨.program ⟨257⟩, ⟨17905⟩⟩
def rawTerms : List Term := Proof.Events387.exact99179RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 99179
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99179.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 99176) (rightBinding := 99177)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16117⟩) (rightExpression := ⟨17902⟩)
    (transferEvent := 99178)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99175.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99160.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99179

namespace SemanticResult99188
def owner : Owner := ⟨.program ⟨257⟩, ⟨16699⟩⟩
def rawTerms : List Term := Proof.Events387.exact99188RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 99188
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99188.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99023.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge99023.frameStart)
    (owner := owner) (leftOwner := SemanticResult90620.owner)
    (rightOwner := SemanticResult99017.owner)
    (leftResult := 90620) (rightResult := 99017)
    (leftActual := SemanticResult90620.actual selector witness)
    (rightActual := SemanticResult99017.actual selector witness)
    (leftRaw := SemanticResult90620.rawTerms)
    (rightRaw := SemanticResult99017.rawTerms)
    (working := LeftOperatorMerge99023.working)
    (leftBinding := 99018) (rightBinding := 99019)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨9944⟩) (rightExpression := ⟨16698⟩)
    (coefficientTransfer := 99020) (summaryTransfer := 99022)
    (rightCoefficientProducer := 99016)
    (rightSummaryTransfer := 99021)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge99023.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound99016.actual selector witness)
    (summaryMagnitude := LeftBound99022.actual selector witness)
    (reconstruction := LeftOperatorMerge99023.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult90620.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99017.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99016.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound99016.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge99023.operationAgreement
  · exact LeftBound99022.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge99023.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 99183 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16115⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16115⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge99023.working
    [{ coefficient := (1), key := LeftRelationMerge99183.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge99183.frameStart
      LeftRelationMerge99183.owner (.relation 99183) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge99183.deltas
    rows := LeftRelationMerge99183.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge99023.working LeftRelationMerge99183.source
        (relationContext LeftRelationMerge99183.source
          LeftRelationMerge99183.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge99023.working, LeftRelationMerge99183.deltas,
    LeftRelationMerge99183.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 99183)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨16699⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge99023.working) (working := relationWorking0)
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
end SemanticResult99188

namespace SemanticResult99195
def owner : Owner := ⟨.program ⟨257⟩, ⟨17904⟩⟩
def rawTerms : List Term := Proof.Events387.exact99195RawTerms
def summary : Bound := (.finite 32188807212483706889510625476608)
def resultEvent : Nat := 99195
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99195.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge99192.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99188.owner)
    (rightOwner := SemanticResult99010.owner)
    (leftResult := 99188) (rightResult := 99010)
    (leftActual := SemanticResult99188.actual selector witness)
    (rightActual := SemanticResult99010.actual selector witness)
    (leftRaw := SemanticResult99188.rawTerms)
    (rightRaw := SemanticResult99010.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 32188807212483504816668771614720) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 99189) (rightBinding := 99190)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨16699⟩) (rightExpression := ⟨17903⟩)
    (coefficientTransfer := 99191) (summaryTransfer := 99194)
    (base := LeftOperatorMerge99192.base)
    (reconstruction := LeftOperatorMerge99192.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99188.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult99010.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge99192.operationAgreement
  · rfl
  · decide
end SemanticResult99195

namespace SemanticResult99200
def owner : Owner := ⟨.program ⟨257⟩, ⟨20811⟩⟩
def rawTerms : List Term := Proof.Events387.exact99200RawTerms
def summary : Bound := (.finite 64377712650190257467641695830016)
def resultEvent : Nat := 99200
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99200.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99195.owner)
    (rightOwner := SemanticResult98713.owner)
    (leftResult := 99195) (rightResult := 98713)
    (leftActual := SemanticResult99195.actual selector witness)
    (rightActual := SemanticResult98713.actual selector witness)
    (leftRaw := SemanticResult99195.rawTerms)
    (rightRaw := SemanticResult98713.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 32188807212483706889510625476608)
    (rightMaximum := 32188905437706550578131070353408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 99196) (rightBinding := 99197)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨17904⟩) (rightExpression := ⟨20810⟩)
    (transferEvent := 99198) (summaryTransferEvent := 99199)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99195.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult98713.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99200

namespace SemanticResult99205
def owner : Owner := ⟨.program ⟨257⟩, ⟨24031⟩⟩
def rawTerms : List Term := Proof.Events387.exact99205RawTerms
def summary : Bound := (.finite 96566716313119651734393211060224)
def resultEvent : Nat := 99205
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99205.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99200.owner)
    (rightOwner := SemanticResult98231.owner)
    (leftResult := 99200) (rightResult := 98231)
    (leftActual := SemanticResult99200.actual selector witness)
    (rightActual := SemanticResult98231.actual selector witness)
    (leftRaw := SemanticResult99200.rawTerms)
    (rightRaw := SemanticResult98231.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 64377712650190257467641695830016)
    (rightMaximum := 32189003662929394266751515230208) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 99201) (rightBinding := 99202)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨20811⟩) (rightExpression := ⟨24030⟩)
    (transferEvent := 99203) (summaryTransferEvent := 99204)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99200.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult98231.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99205

namespace SemanticResult99210
def owner : Owner := ⟨.program ⟨257⟩, ⟨34051⟩⟩
def rawTerms : List Term := Proof.Events387.exact99210RawTerms
def summary : Bound := (.finite 128755916426494733378385616044032)
def resultEvent : Nat := 99210
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99210.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99205.owner)
    (rightOwner := SemanticResult97749.owner)
    (leftResult := 99205) (rightResult := 97749)
    (leftActual := SemanticResult99205.actual selector witness)
    (rightActual := SemanticResult97749.actual selector witness)
    (leftRaw := SemanticResult99205.rawTerms)
    (rightRaw := SemanticResult97749.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 96566716313119651734393211060224)
    (rightMaximum := 32189200113375081643992404983808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 99206) (rightBinding := 99207)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨24031⟩) (rightExpression := ⟨34050⟩)
    (transferEvent := 99208) (summaryTransferEvent := 99209)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99205.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult97749.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99210

namespace SemanticResult99215
def owner : Owner := ⟨.program ⟨257⟩, ⟨53111⟩⟩
def rawTerms : List Term := Proof.Events387.exact99215RawTerms
def summary : Bound := (.finite 160945509440761189776859800535040)
def resultEvent : Nat := 99215
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99215.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99210.owner)
    (rightOwner := SemanticResult97267.owner)
    (leftResult := 99210) (rightResult := 97267)
    (leftActual := SemanticResult99210.actual selector witness)
    (rightActual := SemanticResult97267.actual selector witness)
    (leftRaw := SemanticResult99210.rawTerms)
    (rightRaw := SemanticResult97267.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 128755916426494733378385616044032)
    (rightMaximum := 32189593014266456398474184491008) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 99211) (rightBinding := 99212)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨34051⟩) (rightExpression := ⟨53110⟩)
    (transferEvent := 99213) (summaryTransferEvent := 99214)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99210.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult97267.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99215

namespace SemanticResult99220
def owner : Owner := ⟨.program ⟨257⟩, ⟨56091⟩⟩
def rawTerms : List Term := Proof.Events387.exact99220RawTerms
def summary : Bound := (.finite 193135298905473333552574874779648)
def resultEvent : Nat := 99220
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99220.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99215.owner)
    (rightOwner := SemanticResult96785.owner)
    (leftResult := 99215) (rightResult := 96785)
    (leftActual := SemanticResult99215.actual selector witness)
    (rightActual := SemanticResult96785.actual selector witness)
    (leftRaw := SemanticResult99215.rawTerms)
    (rightRaw := SemanticResult96785.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 160945509440761189776859800535040)
    (rightMaximum := 32189789464712143775715074244608) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 99216) (rightBinding := 99217)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨53111⟩) (rightExpression := ⟨56090⟩)
    (transferEvent := 99218) (summaryTransferEvent := 99219)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99215.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96785.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99220

namespace SemanticResult99225
def owner : Owner := ⟨.program ⟨257⟩, ⟨59071⟩⟩
def rawTerms : List Term := Proof.Events387.exact99225RawTerms
def summary : Bound := (.finite 225325481271076852082771728531456)
def resultEvent : Nat := 99225
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99225.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99220.owner)
    (rightOwner := SemanticResult96303.owner)
    (leftResult := 99220) (rightResult := 96303)
    (leftActual := SemanticResult99220.actual selector witness)
    (rightActual := SemanticResult96303.actual selector witness)
    (leftRaw := SemanticResult99220.rawTerms)
    (rightRaw := SemanticResult96303.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 193135298905473333552574874779648)
    (rightMaximum := 32190182365603518530196853751808) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 99221) (rightBinding := 99222)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨56091⟩) (rightExpression := ⟨59070⟩)
    (transferEvent := 99223) (summaryTransferEvent := 99224)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99220.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult96303.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99225

namespace SemanticResult99230
def owner : Owner := ⟨.program ⟨257⟩, ⟨62051⟩⟩
def rawTerms : List Term := Proof.Events387.exact99230RawTerms
def summary : Bound := (.finite 257515860087126057990209472036864)
def resultEvent : Nat := 99230
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult99230.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddNoMergeClaim
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := 0)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult99225.owner)
    (rightOwner := SemanticResult95821.owner)
    (leftResult := 99225) (rightResult := 95821)
    (leftActual := SemanticResult99225.actual selector witness)
    (rightActual := SemanticResult95821.actual selector witness)
    (leftRaw := SemanticResult99225.rawTerms)
    (rightRaw := SemanticResult95821.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 225325481271076852082771728531456)
    (rightMaximum := 32190378816049205907437743505408) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 99226) (rightBinding := 99227)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨59071⟩) (rightExpression := ⟨62050⟩)
    (transferEvent := 99228) (summaryTransferEvent := 99229)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult99225.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult95821.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult99230

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
