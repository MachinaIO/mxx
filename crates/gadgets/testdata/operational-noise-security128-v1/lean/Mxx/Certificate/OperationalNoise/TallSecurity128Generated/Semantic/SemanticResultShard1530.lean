import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Cert.ResidualShard1530
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBound
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticMergeTree
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard122
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1458
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1460
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1525
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1527
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticResultShard1529

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics

namespace SemanticResult217636
def owner : Owner := ⟨.program ⟨257⟩, ⟨69090⟩⟩
def rawTerms : List Term := Proof.Events850.exact217636RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217636
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217636.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217633) (rightBinding := 217634)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7325⟩) (rightExpression := ⟨69089⟩)
    (transferEvent := 217635)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217632.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217510.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217636

namespace SemanticResult217712
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def rawTerms : List Term := Proof.Events850.exact217712RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217712
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217712.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge217640.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge217640.frameStart)
    (transferEvent := 217639) (owner := owner)
    (leftResult := 217636) (rightResult := 217477)
    (working := LeftOperatorMerge217640.working)
    (reconstruction := LeftOperatorMerge217640.reconstruction)
    (leftReference := .predecessor 0 217637 .coefficient) (rightReference := .predecessor 1 217638 .coefficient)
    (facts := ⟨false, false, none, none, none⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult217636.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217477.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge217640.operationAgreement
  · decide
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217659 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge217640.working
    [{ coefficient := (-1), key := LeftRelationMerge217659.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge217659.frameStart
      LeftRelationMerge217659.owner (.relation 217659) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge217659.deltas
    rows := LeftRelationMerge217659.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge217640.working LeftRelationMerge217659.source
        (relationContext LeftRelationMerge217659.source
          LeftRelationMerge217659.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge217640.working, LeftRelationMerge217659.deltas,
    LeftRelationMerge217659.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217659)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge217640.working) (working := relationWorking0)
    (reconstruction := relationReconstruction0)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt0 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact mergeClaim selector selectorLower selectorUpper witness
  · exact relationAgreement0
  · decide +kernel
theorem relationApplicationAt1 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217662 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking1 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }]
def relationRhsRaw1 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase1 : Polynomial Owner :=
  subtract relationWorking0
    [{ coefficient := (-1), key := LeftRelationMerge217662.source }]
def relationReconstruction1 :
    MergeReconstructionAt history LeftRelationMerge217662.frameStart
      LeftRelationMerge217662.owner (.relation 217662) relationBase1
      relationWorking1 :=
  { deltas := LeftRelationMerge217662.deltas
    rows := LeftRelationMerge217662.rows
    agreement := by decide +kernel }
theorem relationAgreement1 :
    CanonicalAgreement (add relationBase1 relationReconstruction1.deltas)
      (relationPoly relationWorking0 LeftRelationMerge217662.source
        (relationContext LeftRelationMerge217662.source
          LeftRelationMerge217662.source.centralFactors 0 2) (-1)
        (relationRhsRaw1.map Term.toExact)) := by
  dsimp [relationReconstruction1, relationBase1, relationWorking1,
    relationRhsRaw1, relationWorking0, LeftRelationMerge217662.deltas,
    LeftRelationMerge217662.source]
  decide +kernel
theorem relationClaim1 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking1 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217662)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw1)
    (accumulator := relationWorking0) (working := relationWorking1)
    (reconstruction := relationReconstruction1)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt1 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim0 selector selectorLower selectorUpper witness
  · exact relationAgreement1
  · decide +kernel
theorem relationApplicationAt2 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217665 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking2 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }]
def relationRhsRaw2 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase2 : Polynomial Owner :=
  subtract relationWorking1
    [{ coefficient := (-1), key := LeftRelationMerge217665.source }]
def relationReconstruction2 :
    MergeReconstructionAt history LeftRelationMerge217665.frameStart
      LeftRelationMerge217665.owner (.relation 217665) relationBase2
      relationWorking2 :=
  { deltas := LeftRelationMerge217665.deltas
    rows := LeftRelationMerge217665.rows
    agreement := by decide +kernel }
theorem relationAgreement2 :
    CanonicalAgreement (add relationBase2 relationReconstruction2.deltas)
      (relationPoly relationWorking1 LeftRelationMerge217665.source
        (relationContext LeftRelationMerge217665.source
          LeftRelationMerge217665.source.centralFactors 0 2) (-1)
        (relationRhsRaw2.map Term.toExact)) := by
  dsimp [relationReconstruction2, relationBase2, relationWorking2,
    relationRhsRaw2, relationWorking1, LeftRelationMerge217665.deltas,
    LeftRelationMerge217665.source]
  decide +kernel
theorem relationClaim2 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking2 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217665)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw2)
    (accumulator := relationWorking1) (working := relationWorking2)
    (reconstruction := relationReconstruction2)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt2 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim1 selector selectorLower selectorUpper witness
  · exact relationAgreement2
  · decide +kernel
theorem relationApplicationAt3 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217668 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking3 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }]
def relationRhsRaw3 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase3 : Polynomial Owner :=
  subtract relationWorking2
    [{ coefficient := (-1), key := LeftRelationMerge217668.source }]
def relationReconstruction3 :
    MergeReconstructionAt history LeftRelationMerge217668.frameStart
      LeftRelationMerge217668.owner (.relation 217668) relationBase3
      relationWorking3 :=
  { deltas := LeftRelationMerge217668.deltas
    rows := LeftRelationMerge217668.rows
    agreement := by decide +kernel }
theorem relationAgreement3 :
    CanonicalAgreement (add relationBase3 relationReconstruction3.deltas)
      (relationPoly relationWorking2 LeftRelationMerge217668.source
        (relationContext LeftRelationMerge217668.source
          LeftRelationMerge217668.source.centralFactors 0 2) (-1)
        (relationRhsRaw3.map Term.toExact)) := by
  dsimp [relationReconstruction3, relationBase3, relationWorking3,
    relationRhsRaw3, relationWorking2, LeftRelationMerge217668.deltas,
    LeftRelationMerge217668.source]
  decide +kernel
theorem relationClaim3 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking3 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217668)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw3)
    (accumulator := relationWorking2) (working := relationWorking3)
    (reconstruction := relationReconstruction3)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt3 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim2 selector selectorLower selectorUpper witness
  · exact relationAgreement3
  · decide +kernel
theorem relationApplicationAt4 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217671 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking4 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }]
def relationRhsRaw4 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase4 : Polynomial Owner :=
  subtract relationWorking3
    [{ coefficient := (-1), key := LeftRelationMerge217671.source }]
def relationReconstruction4 :
    MergeReconstructionAt history LeftRelationMerge217671.frameStart
      LeftRelationMerge217671.owner (.relation 217671) relationBase4
      relationWorking4 :=
  { deltas := LeftRelationMerge217671.deltas
    rows := LeftRelationMerge217671.rows
    agreement := by decide +kernel }
theorem relationAgreement4 :
    CanonicalAgreement (add relationBase4 relationReconstruction4.deltas)
      (relationPoly relationWorking3 LeftRelationMerge217671.source
        (relationContext LeftRelationMerge217671.source
          LeftRelationMerge217671.source.centralFactors 0 2) (-1)
        (relationRhsRaw4.map Term.toExact)) := by
  dsimp [relationReconstruction4, relationBase4, relationWorking4,
    relationRhsRaw4, relationWorking3, LeftRelationMerge217671.deltas,
    LeftRelationMerge217671.source]
  decide +kernel
theorem relationClaim4 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking4 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217671)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw4)
    (accumulator := relationWorking3) (working := relationWorking4)
    (reconstruction := relationReconstruction4)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt4 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim3 selector selectorLower selectorUpper witness
  · exact relationAgreement4
  · decide +kernel
theorem relationApplicationAt5 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217674 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking5 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }]
def relationRhsRaw5 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase5 : Polynomial Owner :=
  subtract relationWorking4
    [{ coefficient := (-1), key := LeftRelationMerge217674.source }]
def relationReconstruction5 :
    MergeReconstructionAt history LeftRelationMerge217674.frameStart
      LeftRelationMerge217674.owner (.relation 217674) relationBase5
      relationWorking5 :=
  { deltas := LeftRelationMerge217674.deltas
    rows := LeftRelationMerge217674.rows
    agreement := by decide +kernel }
theorem relationAgreement5 :
    CanonicalAgreement (add relationBase5 relationReconstruction5.deltas)
      (relationPoly relationWorking4 LeftRelationMerge217674.source
        (relationContext LeftRelationMerge217674.source
          LeftRelationMerge217674.source.centralFactors 0 2) (-1)
        (relationRhsRaw5.map Term.toExact)) := by
  dsimp [relationReconstruction5, relationBase5, relationWorking5,
    relationRhsRaw5, relationWorking4, LeftRelationMerge217674.deltas,
    LeftRelationMerge217674.source]
  decide +kernel
theorem relationClaim5 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking5 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217674)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw5)
    (accumulator := relationWorking4) (working := relationWorking5)
    (reconstruction := relationReconstruction5)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt5 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim4 selector selectorLower selectorUpper witness
  · exact relationAgreement5
  · decide +kernel
theorem relationApplicationAt6 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217677 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking6 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }]
def relationRhsRaw6 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase6 : Polynomial Owner :=
  subtract relationWorking5
    [{ coefficient := (-1), key := LeftRelationMerge217677.source }]
def relationReconstruction6 :
    MergeReconstructionAt history LeftRelationMerge217677.frameStart
      LeftRelationMerge217677.owner (.relation 217677) relationBase6
      relationWorking6 :=
  { deltas := LeftRelationMerge217677.deltas
    rows := LeftRelationMerge217677.rows
    agreement := by decide +kernel }
theorem relationAgreement6 :
    CanonicalAgreement (add relationBase6 relationReconstruction6.deltas)
      (relationPoly relationWorking5 LeftRelationMerge217677.source
        (relationContext LeftRelationMerge217677.source
          LeftRelationMerge217677.source.centralFactors 0 2) (-1)
        (relationRhsRaw6.map Term.toExact)) := by
  dsimp [relationReconstruction6, relationBase6, relationWorking6,
    relationRhsRaw6, relationWorking5, LeftRelationMerge217677.deltas,
    LeftRelationMerge217677.source]
  decide +kernel
theorem relationClaim6 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking6 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217677)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨29299⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw6)
    (accumulator := relationWorking5) (working := relationWorking6)
    (reconstruction := relationReconstruction6)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt6 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim5 selector selectorLower selectorUpper witness
  · exact relationAgreement6
  · decide +kernel
theorem relationApplicationAt7 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217680 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking7 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }]
def relationRhsRaw7 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase7 : Polynomial Owner :=
  subtract relationWorking6
    [{ coefficient := (-1), key := LeftRelationMerge217680.source }]
def relationReconstruction7 :
    MergeReconstructionAt history LeftRelationMerge217680.frameStart
      LeftRelationMerge217680.owner (.relation 217680) relationBase7
      relationWorking7 :=
  { deltas := LeftRelationMerge217680.deltas
    rows := LeftRelationMerge217680.rows
    agreement := by decide +kernel }
theorem relationAgreement7 :
    CanonicalAgreement (add relationBase7 relationReconstruction7.deltas)
      (relationPoly relationWorking6 LeftRelationMerge217680.source
        (relationContext LeftRelationMerge217680.source
          LeftRelationMerge217680.source.centralFactors 0 2) (-1)
        (relationRhsRaw7.map Term.toExact)) := by
  dsimp [relationReconstruction7, relationBase7, relationWorking7,
    relationRhsRaw7, relationWorking6, LeftRelationMerge217680.deltas,
    LeftRelationMerge217680.source]
  decide +kernel
theorem relationClaim7 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking7 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217680)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨26619⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw7)
    (accumulator := relationWorking6) (working := relationWorking7)
    (reconstruction := relationReconstruction7)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt7 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim6 selector selectorLower selectorUpper witness
  · exact relationAgreement7
  · decide +kernel
theorem relationApplicationAt8 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217683 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking8 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw8 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase8 : Polynomial Owner :=
  subtract relationWorking7
    [{ coefficient := (-1), key := LeftRelationMerge217683.source }]
def relationReconstruction8 :
    MergeReconstructionAt history LeftRelationMerge217683.frameStart
      LeftRelationMerge217683.owner (.relation 217683) relationBase8
      relationWorking8 :=
  { deltas := LeftRelationMerge217683.deltas
    rows := LeftRelationMerge217683.rows
    agreement := by decide +kernel }
theorem relationAgreement8 :
    CanonicalAgreement (add relationBase8 relationReconstruction8.deltas)
      (relationPoly relationWorking7 LeftRelationMerge217683.source
        (relationContext LeftRelationMerge217683.source
          LeftRelationMerge217683.source.centralFactors 0 2) (-1)
        (relationRhsRaw8.map Term.toExact)) := by
  dsimp [relationReconstruction8, relationBase8, relationWorking8,
    relationRhsRaw8, relationWorking7, LeftRelationMerge217683.deltas,
    LeftRelationMerge217683.source]
  decide +kernel
theorem relationClaim8 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking8 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217683)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨66601⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw8)
    (accumulator := relationWorking7) (working := relationWorking8)
    (reconstruction := relationReconstruction8)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt8 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim7 selector selectorLower selectorUpper witness
  · exact relationAgreement8
  · decide +kernel
theorem relationApplicationAt9 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217686 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking9 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw9 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase9 : Polynomial Owner :=
  subtract relationWorking8
    [{ coefficient := (-1), key := LeftRelationMerge217686.source }]
def relationReconstruction9 :
    MergeReconstructionAt history LeftRelationMerge217686.frameStart
      LeftRelationMerge217686.owner (.relation 217686) relationBase9
      relationWorking9 :=
  { deltas := LeftRelationMerge217686.deltas
    rows := LeftRelationMerge217686.rows
    agreement := by decide +kernel }
theorem relationAgreement9 :
    CanonicalAgreement (add relationBase9 relationReconstruction9.deltas)
      (relationPoly relationWorking8 LeftRelationMerge217686.source
        (relationContext LeftRelationMerge217686.source
          LeftRelationMerge217686.source.centralFactors 0 2) (-1)
        (relationRhsRaw9.map Term.toExact)) := by
  dsimp [relationReconstruction9, relationBase9, relationWorking9,
    relationRhsRaw9, relationWorking8, LeftRelationMerge217686.deltas,
    LeftRelationMerge217686.source]
  decide +kernel
theorem relationClaim9 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking9 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217686)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨63081⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw9)
    (accumulator := relationWorking8) (working := relationWorking9)
    (reconstruction := relationReconstruction9)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt9 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim8 selector selectorLower selectorUpper witness
  · exact relationAgreement9
  · decide +kernel
theorem relationApplicationAt10 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217689 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking10 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw10 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase10 : Polynomial Owner :=
  subtract relationWorking9
    [{ coefficient := (-1), key := LeftRelationMerge217689.source }]
def relationReconstruction10 :
    MergeReconstructionAt history LeftRelationMerge217689.frameStart
      LeftRelationMerge217689.owner (.relation 217689) relationBase10
      relationWorking10 :=
  { deltas := LeftRelationMerge217689.deltas
    rows := LeftRelationMerge217689.rows
    agreement := by decide +kernel }
theorem relationAgreement10 :
    CanonicalAgreement (add relationBase10 relationReconstruction10.deltas)
      (relationPoly relationWorking9 LeftRelationMerge217689.source
        (relationContext LeftRelationMerge217689.source
          LeftRelationMerge217689.source.centralFactors 0 2) (-1)
        (relationRhsRaw10.map Term.toExact)) := by
  dsimp [relationReconstruction10, relationBase10, relationWorking10,
    relationRhsRaw10, relationWorking9, LeftRelationMerge217689.deltas,
    LeftRelationMerge217689.source]
  decide +kernel
theorem relationClaim10 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking10 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217689)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw10)
    (accumulator := relationWorking9) (working := relationWorking10)
    (reconstruction := relationReconstruction10)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt10 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim9 selector selectorLower selectorUpper witness
  · exact relationAgreement10
  · decide +kernel
theorem relationApplicationAt11 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217692 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking11 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw11 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase11 : Polynomial Owner :=
  subtract relationWorking10
    [{ coefficient := (-1), key := LeftRelationMerge217692.source }]
def relationReconstruction11 :
    MergeReconstructionAt history LeftRelationMerge217692.frameStart
      LeftRelationMerge217692.owner (.relation 217692) relationBase11
      relationWorking11 :=
  { deltas := LeftRelationMerge217692.deltas
    rows := LeftRelationMerge217692.rows
    agreement := by decide +kernel }
theorem relationAgreement11 :
    CanonicalAgreement (add relationBase11 relationReconstruction11.deltas)
      (relationPoly relationWorking10 LeftRelationMerge217692.source
        (relationContext LeftRelationMerge217692.source
          LeftRelationMerge217692.source.centralFactors 0 2) (-1)
        (relationRhsRaw11.map Term.toExact)) := by
  dsimp [relationReconstruction11, relationBase11, relationWorking11,
    relationRhsRaw11, relationWorking10, LeftRelationMerge217692.deltas,
    LeftRelationMerge217692.source]
  decide +kernel
theorem relationClaim11 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking11 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217692)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨57121⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw11)
    (accumulator := relationWorking10) (working := relationWorking11)
    (reconstruction := relationReconstruction11)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt11 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim10 selector selectorLower selectorUpper witness
  · exact relationAgreement11
  · decide +kernel
theorem relationApplicationAt12 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217695 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking12 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw12 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase12 : Polynomial Owner :=
  subtract relationWorking11
    [{ coefficient := (-1), key := LeftRelationMerge217695.source }]
def relationReconstruction12 :
    MergeReconstructionAt history LeftRelationMerge217695.frameStart
      LeftRelationMerge217695.owner (.relation 217695) relationBase12
      relationWorking12 :=
  { deltas := LeftRelationMerge217695.deltas
    rows := LeftRelationMerge217695.rows
    agreement := by decide +kernel }
theorem relationAgreement12 :
    CanonicalAgreement (add relationBase12 relationReconstruction12.deltas)
      (relationPoly relationWorking11 LeftRelationMerge217695.source
        (relationContext LeftRelationMerge217695.source
          LeftRelationMerge217695.source.centralFactors 0 2) (-1)
        (relationRhsRaw12.map Term.toExact)) := by
  dsimp [relationReconstruction12, relationBase12, relationWorking12,
    relationRhsRaw12, relationWorking11, LeftRelationMerge217695.deltas,
    LeftRelationMerge217695.source]
  decide +kernel
theorem relationClaim12 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking12 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217695)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨54141⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw12)
    (accumulator := relationWorking11) (working := relationWorking12)
    (reconstruction := relationReconstruction12)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt12 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim11 selector selectorLower selectorUpper witness
  · exact relationAgreement12
  · decide +kernel
theorem relationApplicationAt13 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217698 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking13 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw13 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase13 : Polynomial Owner :=
  subtract relationWorking12
    [{ coefficient := (-1), key := LeftRelationMerge217698.source }]
def relationReconstruction13 :
    MergeReconstructionAt history LeftRelationMerge217698.frameStart
      LeftRelationMerge217698.owner (.relation 217698) relationBase13
      relationWorking13 :=
  { deltas := LeftRelationMerge217698.deltas
    rows := LeftRelationMerge217698.rows
    agreement := by decide +kernel }
theorem relationAgreement13 :
    CanonicalAgreement (add relationBase13 relationReconstruction13.deltas)
      (relationPoly relationWorking12 LeftRelationMerge217698.source
        (relationContext LeftRelationMerge217698.source
          LeftRelationMerge217698.source.centralFactors 0 2) (-1)
        (relationRhsRaw13.map Term.toExact)) := by
  dsimp [relationReconstruction13, relationBase13, relationWorking13,
    relationRhsRaw13, relationWorking12, LeftRelationMerge217698.deltas,
    LeftRelationMerge217698.source]
  decide +kernel
theorem relationClaim13 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking13 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217698)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨51161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw13)
    (accumulator := relationWorking12) (working := relationWorking13)
    (reconstruction := relationReconstruction13)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt13 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim12 selector selectorLower selectorUpper witness
  · exact relationAgreement13
  · decide +kernel
theorem relationApplicationAt14 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217701 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking14 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw14 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase14 : Polynomial Owner :=
  subtract relationWorking13
    [{ coefficient := (-1), key := LeftRelationMerge217701.source }]
def relationReconstruction14 :
    MergeReconstructionAt history LeftRelationMerge217701.frameStart
      LeftRelationMerge217701.owner (.relation 217701) relationBase14
      relationWorking14 :=
  { deltas := LeftRelationMerge217701.deltas
    rows := LeftRelationMerge217701.rows
    agreement := by decide +kernel }
theorem relationAgreement14 :
    CanonicalAgreement (add relationBase14 relationReconstruction14.deltas)
      (relationPoly relationWorking13 LeftRelationMerge217701.source
        (relationContext LeftRelationMerge217701.source
          LeftRelationMerge217701.source.centralFactors 0 2) (-1)
        (relationRhsRaw14.map Term.toExact)) := by
  dsimp [relationReconstruction14, relationBase14, relationWorking14,
    relationRhsRaw14, relationWorking13, LeftRelationMerge217701.deltas,
    LeftRelationMerge217701.source]
  decide +kernel
theorem relationClaim14 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking14 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217701)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw14)
    (accumulator := relationWorking13) (working := relationWorking14)
    (reconstruction := relationReconstruction14)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt14 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim13 selector selectorLower selectorUpper witness
  · exact relationAgreement14
  · decide +kernel
theorem relationApplicationAt15 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217704 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking15 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw15 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase15 : Polynomial Owner :=
  subtract relationWorking14
    [{ coefficient := (-1), key := LeftRelationMerge217704.source }]
def relationReconstruction15 :
    MergeReconstructionAt history LeftRelationMerge217704.frameStart
      LeftRelationMerge217704.owner (.relation 217704) relationBase15
      relationWorking15 :=
  { deltas := LeftRelationMerge217704.deltas
    rows := LeftRelationMerge217704.rows
    agreement := by decide +kernel }
theorem relationAgreement15 :
    CanonicalAgreement (add relationBase15 relationReconstruction15.deltas)
      (relationPoly relationWorking14 LeftRelationMerge217704.source
        (relationContext LeftRelationMerge217704.source
          LeftRelationMerge217704.source.centralFactors 0 2) (-1)
        (relationRhsRaw15.map Term.toExact)) := by
  dsimp [relationReconstruction15, relationBase15, relationWorking15,
    relationRhsRaw15, relationWorking14, LeftRelationMerge217704.deltas,
    LeftRelationMerge217704.source]
  decide +kernel
theorem relationClaim15 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking15 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217704)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw15)
    (accumulator := relationWorking14) (working := relationWorking15)
    (reconstruction := relationReconstruction15)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt15 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim14 selector selectorLower selectorUpper witness
  · exact relationAgreement15
  · decide +kernel
theorem relationApplicationAt16 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217707 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking16 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw16 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase16 : Polynomial Owner :=
  subtract relationWorking15
    [{ coefficient := (-1), key := LeftRelationMerge217707.source }]
def relationReconstruction16 :
    MergeReconstructionAt history LeftRelationMerge217707.frameStart
      LeftRelationMerge217707.owner (.relation 217707) relationBase16
      relationWorking16 :=
  { deltas := LeftRelationMerge217707.deltas
    rows := LeftRelationMerge217707.rows
    agreement := by decide +kernel }
theorem relationAgreement16 :
    CanonicalAgreement (add relationBase16 relationReconstruction16.deltas)
      (relationPoly relationWorking15 LeftRelationMerge217707.source
        (relationContext LeftRelationMerge217707.source
          LeftRelationMerge217707.source.centralFactors 0 2) (-1)
        (relationRhsRaw16.map Term.toExact)) := by
  dsimp [relationReconstruction16, relationBase16, relationWorking16,
    relationRhsRaw16, relationWorking15, LeftRelationMerge217707.deltas,
    LeftRelationMerge217707.source]
  decide +kernel
theorem relationClaim16 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking16 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217707)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw16)
    (accumulator := relationWorking15) (working := relationWorking16)
    (reconstruction := relationReconstruction16)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt16 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim15 selector selectorLower selectorUpper witness
  · exact relationAgreement16
  · decide +kernel
theorem relationApplicationAt17 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217710 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking17 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationRhsRaw17 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }]
def relationBase17 : Polynomial Owner :=
  subtract relationWorking16
    [{ coefficient := (-1), key := LeftRelationMerge217710.source }]
def relationReconstruction17 :
    MergeReconstructionAt history LeftRelationMerge217710.frameStart
      LeftRelationMerge217710.owner (.relation 217710) relationBase17
      relationWorking17 :=
  { deltas := LeftRelationMerge217710.deltas
    rows := LeftRelationMerge217710.rows
    agreement := by decide +kernel }
theorem relationAgreement17 :
    CanonicalAgreement (add relationBase17 relationReconstruction17.deltas)
      (relationPoly relationWorking16 LeftRelationMerge217710.source
        (relationContext LeftRelationMerge217710.source
          LeftRelationMerge217710.source.centralFactors 0 2) (-1)
        (relationRhsRaw17.map Term.toExact)) := by
  dsimp [relationReconstruction17, relationBase17, relationWorking17,
    relationRhsRaw17, relationWorking16, LeftRelationMerge217710.deltas,
    LeftRelationMerge217710.source]
  decide +kernel
theorem relationClaim17 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking17 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217710)
    (frameStart := 216961) (owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw17)
    (accumulator := relationWorking16) (working := relationWorking17)
    (reconstruction := relationReconstruction17)
    (actual := actual selector witness) (summary := summary)
  · exact relationApplicationAt17 selector selectorLower selectorUpper
  · rfl
  · rfl
  · decide +kernel
  · exact relationClaim16 selector selectorLower selectorUpper witness
  · exact relationAgreement17
  · decide +kernel
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply exactClaimAt_of_mergeClaim
    (relationClaim17 selector selectorLower selectorUpper witness)
  · decide +kernel
  · rfl
end SemanticResult217712

namespace SemanticResult217715
def owner : Owner := ⟨.program ⟨257⟩, ⟨67457⟩⟩
def rawTerms : List Term := Proof.Events850.exact217715RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217715
def producerEvent : Nat := 217714
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217715.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 216961, .finite 18, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult217715

namespace SemanticResult217720
def owner : Owner := ⟨.program ⟨257⟩, ⟨67459⟩⟩
def rawTerms : List Term := Proof.Events850.exact217720RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217720
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217720.actual selector witness
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge217719.working .exactZero) := by
  apply operatorProductMergeClaim
    (document := document) (history := history) (env := witness.env)
    (frameStart := LeftOperatorMerge217719.frameStart)
    (transferEvent := 217718) (owner := owner)
    (leftResult := 217488) (rightResult := 217715)
    (working := LeftOperatorMerge217719.working)
    (reconstruction := LeftOperatorMerge217719.reconstruction)
    (leftReference := .predecessor 0 217716 .coefficient) (rightReference := .predecessor 1 217717 .coefficient)
    (facts := ⟨false, true, none, none, some 1⟩)
    (leftScalar := false) (rightScalar := false)
  · rfl
  · rfl
  · exact SemanticResult217488.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217715.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge217719.operationAgreement
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
end SemanticResult217720

namespace SemanticResult217723
def owner : Owner := ⟨.program ⟨257⟩, ⟨7233⟩⟩
def rawTerms : List Term := Proof.Events850.exact217723RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217723
def producerEvent : Nat := 217722
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217723.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.operator), 216961, .large, .authorityOperator, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult217723

namespace SemanticResult217727
def owner : Owner := ⟨.program ⟨257⟩, ⟨67464⟩⟩
def rawTerms : List Term := Proof.Events850.exact217727RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217727
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217727.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217724) (rightBinding := 217725)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨7233⟩) (rightExpression := ⟨67459⟩)
    (transferEvent := 217726)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217723.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217720.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217727

namespace SemanticResult217731
def owner : Owner := ⟨.program ⟨257⟩, ⟨71241⟩⟩
def rawTerms : List Term := Proof.Events850.exact217731RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217731
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217731.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorSubNoMergeExactZeroClaimAt
    (document := document) (history := history) (env := witness.env)
    (leftBinding := 217728) (rightBinding := 217729)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨67464⟩) (rightExpression := ⟨71237⟩)
    (transferEvent := 217730)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217727.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217712.claimSound selector selectorLower selectorUpper witness
  · rfl
  · decide +kernel
  · decide
end SemanticResult217731

namespace SemanticResult217774
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def rawTerms : List Term := Proof.Events850.exact217774RawTerms
def summary : Bound := (.finite 202072841853861888)
def resultEvent : Nat := 217774
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217774.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1376256
      (.finite ⟨26, by decide⟩)
      (.finite ⟨5647228698, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge216371.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge216371.frameStart)
    (owner := owner) (leftOwner := SemanticResult207620.owner)
    (rightOwner := SemanticResult216365.owner)
    (leftResult := 207620) (rightResult := 216365)
    (leftActual := SemanticResult207620.actual selector witness)
    (rightActual := SemanticResult216365.actual selector witness)
    (leftRaw := SemanticResult207620.rawTerms)
    (rightRaw := SemanticResult216365.rawTerms)
    (working := LeftOperatorMerge216371.working)
    (leftBinding := 216366) (rightBinding := 216367)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨5599⟩) (rightExpression := ⟨68372⟩)
    (coefficientTransfer := 216368) (summaryTransfer := 216370)
    (rightCoefficientProducer := 216364)
    (rightSummaryTransfer := 216369)
    (leftMaximum := ⟨26, by decide⟩)
    (rightProducerMaximum := ⟨5647228698, by decide⟩)
    (rightRecordedMaximum := 5647228698)
    (rightSummaryMaximum := ⟨5647228698, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1376256)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge216371.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound216364.actual selector witness)
    (summaryMagnitude := LeftBound216370.actual selector witness)
    (reconstruction := LeftOperatorMerge216371.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207620.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult216365.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216364.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound216364.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge216371.operationAgreement
  · exact LeftBound216370.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge216371.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217735 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26619⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29299⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54141⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57121⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63081⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66601⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }, { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge216371.working
    [{ coefficient := (1), key := LeftRelationMerge217735.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge217735.frameStart
      LeftRelationMerge217735.owner (.relation 217735) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge217735.deltas
    rows := LeftRelationMerge217735.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge216371.working LeftRelationMerge217735.source
        (relationContext LeftRelationMerge217735.source
          LeftRelationMerge217735.source.centralFactors 0 2) (1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge216371.working, LeftRelationMerge217735.deltas,
    LeftRelationMerge217735.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply universalRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217735)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩)
    (outerCoefficient := 1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge216371.working) (working := relationWorking0)
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
end SemanticResult217774

namespace SemanticResult217815
def owner : Owner := ⟨.program ⟨257⟩, ⟨71239⟩⟩
def rawTerms : List Term := Proof.Events850.exact217815RawTerms
def summary : Bound := (.finite 6221717896068416040249469506489977540968448)
def resultEvent : Nat := 217815
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217815.actual selector witness
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  apply operatorAddFiniteMergeClaimAt
    (document := document) (history := history) (env := witness.env)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (frameStart := LeftOperatorMerge217778.frameStart)
    (resultEvent := resultEvent) (owner := owner)
    (leftOwner := SemanticResult217774.owner)
    (rightOwner := SemanticResult216358.owner)
    (leftResult := 217774) (rightResult := 216358)
    (leftActual := SemanticResult217774.actual selector witness)
    (rightActual := SemanticResult216358.actual selector witness)
    (leftRaw := SemanticResult217774.rawTerms)
    (rightRaw := SemanticResult216358.rawTerms)
    (outputRaw := rawTerms) (leftMaximum := 202072841853861888)
    (rightMaximum := 6221717896068416040249469304417135687106560) (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40)
    (leftBinding := 217775) (rightBinding := 217776)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨68373⟩) (rightExpression := ⟨71238⟩)
    (coefficientTransfer := 217777) (summaryTransfer := 217814)
    (base := LeftOperatorMerge217778.base)
    (reconstruction := LeftOperatorMerge217778.reconstruction)
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217774.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult216358.claimSound selector selectorLower selectorUpper witness
  · exact LeftOperatorMerge217778.operationAgreement
  · rfl
  · decide
end SemanticResult217815

namespace SemanticResult217825
def owner : Owner := ⟨.program ⟨257⟩, ⟨71240⟩⟩
def rawTerms : List Term := Proof.Events850.exact217825RawTerms
def summary : Bound := (.finite 66805187221379434678483228029309283225584960819691520)
def resultEvent : Nat := 217825
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217825.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨6221717896068416040249469506489977540968448, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge217821.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge217821.frameStart)
    (owner := owner) (leftOwner := SemanticResult217815.owner)
    (rightOwner := SemanticResult15522.owner)
    (leftResult := 217815) (rightResult := 15522)
    (leftActual := SemanticResult217815.actual selector witness)
    (rightActual := SemanticResult15522.actual selector witness)
    (leftRaw := SemanticResult217815.rawTerms)
    (rightRaw := SemanticResult15522.rawTerms)
    (working := LeftOperatorMerge217821.working)
    (leftBinding := 217816) (rightBinding := 217817)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨71239⟩) (rightExpression := ⟨7140⟩)
    (coefficientTransfer := 217818) (summaryTransfer := 217820)
    (rightCoefficientProducer := 15521)
    (rightSummaryTransfer := 217819)
    (leftMaximum := ⟨6221717896068416040249469506489977540968448, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge217821.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftBound15521.actual selector witness)
    (summaryMagnitude := LeftBound217820.actual selector witness)
    (reconstruction := LeftOperatorMerge217821.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult217815.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult15522.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15521.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftBound15521.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge217821.operationAgreement
  · exact LeftBound217820.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge217821.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217823 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge217821.working
    [{ coefficient := (-1), key := LeftRelationMerge217823.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge217823.frameStart
      LeftRelationMerge217823.owner (.relation 217823) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge217823.deltas
    rows := LeftRelationMerge217823.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge217821.working LeftRelationMerge217823.source
        (relationContext LeftRelationMerge217823.source
          LeftRelationMerge217823.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge217821.working, LeftRelationMerge217823.deltas,
    LeftRelationMerge217823.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217823)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨71240⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨67457⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge217821.working) (working := relationWorking0)
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
end SemanticResult217825

namespace SemanticResult217829
def owner : Owner := ⟨.program ⟨257⟩, ⟨49300⟩⟩
def rawTerms : List Term := Proof.Events850.exact217829RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217829
def producerEvent : Nat := 217828
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217829.actual selector witness
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
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult217829

namespace SemanticResult217832
def owner : Owner := ⟨.program ⟨257⟩, ⟨50023⟩⟩
def rawTerms : List Term := Proof.Events850.exact217832RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217832
def producerEvent : Nat := 217831
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217832.actual selector witness
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
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult217832

namespace SemanticResult217842
def owner : Owner := ⟨.program ⟨257⟩, ⟨50025⟩⟩
def rawTerms : List Term := Proof.Events850.exact217842RawTerms
def summary : Bound := (.finite 32194504275408438756654574469120)
def resultEvent : Nat := 217842
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217842.actual selector witness
def computedSummary : Bound :=
  boundOfCoeffClass
    (EventReplay.productWithFactor 1310720
      (.finite ⟨2998346861024241778688, by decide⟩)
      (.finite ⟨8192, by decide⟩))
theorem computedMergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge217838.working computedSummary) := by
  apply operatorProductFiniteMergeClaim
    (document := document) (history := history) (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817)
    (selector := some selector) (witness := witness) (frameStart := LeftOperatorMerge217838.frameStart)
    (owner := owner) (leftOwner := SemanticResult207806.owner)
    (rightOwner := SemanticResult217832.owner)
    (leftResult := 207806) (rightResult := 217832)
    (leftActual := SemanticResult207806.actual selector witness)
    (rightActual := SemanticResult217832.actual selector witness)
    (leftRaw := SemanticResult207806.rawTerms)
    (rightRaw := SemanticResult217832.rawTerms)
    (working := LeftOperatorMerge217838.working)
    (leftBinding := 217833) (rightBinding := 217834)
    (leftInputPosition := 0) (rightInputPosition := 1)
    (leftExpression := ⟨49661⟩) (rightExpression := ⟨50023⟩)
    (coefficientTransfer := 217835) (summaryTransfer := 217837)
    (rightCoefficientProducer := 217831)
    (rightSummaryTransfer := 217836)
    (leftMaximum := ⟨2998346861024241778688, by decide⟩)
    (rightProducerMaximum := ⟨8192, by decide⟩)
    (rightRecordedMaximum := 8192)
    (rightSummaryMaximum := ⟨8192, by decide⟩)
    (leftScalar := false) (rightScalar := false) (factor := 1310720)
    (valueType := .matrix "2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817" 32768 1 40) (base := LeftOperatorMerge217838.base)
    (coefficientFacts := ⟨false, false, none, none, none⟩) (summaryFacts := ⟨false, false, none, none, none⟩)
    (rightMagnitude := LeftAuthority217831.actual selector witness)
    (summaryMagnitude := LeftBound217837.actual selector witness)
    (reconstruction := LeftOperatorMerge217838.reconstruction)
    (rightResultAt := by rfl)
  · rfl
  · rfl
  · rfl
  · rfl
  · exact SemanticResult207806.claimSound selector selectorLower selectorUpper witness
  · exact SemanticResult217832.claimSound selector selectorLower selectorUpper witness
  · exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217831.bound, addKnownList, EventReplay.addKnown, EventReplay.productWithFactor,
        EventReplay.scaleMagnitude, EventReplay.scaleValue, EventReplay.productNonempty,
        RecordedBoundRefines] <;> decide)
      (LeftAuthority217831.derived selector witness)
  · dsimp [RecordedBoundRefines]
    decide
  · decide
  · exact LeftOperatorMerge217838.operationAgreement
  · exact LeftBound217837.derived selector witness
  · decide
  · decide
theorem mergeClaim (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact LeftOperatorMerge217838.working summary) := by
  have claim := computedMergeClaim selector selectorLower selectorUpper witness
  have summaryEq : computedSummary = summary := by
    dsimp [computedSummary, summary, boundOfCoeffClass, EventReplay.productWithFactor]
  simpa only [summaryEq] using claim
theorem relationApplicationAt0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum) :
    RelationApplicationAt document history (some selector) 217840 := by
  refine ⟨_, _, _, _, _, _, _, by rfl, rfl, ?_, by decide⟩
  simp only [selectorMinimum] at selectorLower
  simp only [selectorMaximum] at selectorUpper
  simp [ownerAtSelector, document, selectorLower, selectorUpper]
def relationWorking0 : Polynomial Owner := [{ coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩] } }, { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49300⟩⟩] } }]
def relationRhsRaw0 : List Term := [{ coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49300⟩⟩] } }]
def relationBase0 : Polynomial Owner :=
  subtract LeftOperatorMerge217838.working
    [{ coefficient := (-1), key := LeftRelationMerge217840.source }]
def relationReconstruction0 :
    MergeReconstructionAt history LeftRelationMerge217840.frameStart
      LeftRelationMerge217840.owner (.relation 217840) relationBase0
      relationWorking0 :=
  { deltas := LeftRelationMerge217840.deltas
    rows := LeftRelationMerge217840.rows
    agreement := by decide +kernel }
theorem relationAgreement0 :
    CanonicalAgreement (add relationBase0 relationReconstruction0.deltas)
      (relationPoly LeftOperatorMerge217838.working LeftRelationMerge217840.source
        (relationContext LeftRelationMerge217840.source
          LeftRelationMerge217840.source.centralFactors 0 2) (-1)
        (relationRhsRaw0.map Term.toExact)) := by
  dsimp [relationReconstruction0, relationBase0, relationWorking0,
    relationRhsRaw0, LeftOperatorMerge217838.working, LeftRelationMerge217840.deltas,
    LeftRelationMerge217840.source]
  decide +kernel
theorem relationClaim0 (selector : Nat)
    (selectorLower : selectorMinimum ≤ selector) (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ValueClaim.Interprets 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env (actual selector witness)
      (.exact relationWorking0 summary) := by
  apply gadgetRelationMergeClaim
    (document := document) (history := history) (selector := some selector)
    (modulus := 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) (witness := witness) (application := 217840)
    (frameStart := 0) (owner := ⟨.program ⟨257⟩, ⟨50025⟩⟩)
    (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩) (lhs := ⟨[], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50023⟩⟩]⟩)
    (outerCoefficient := -1) (orderedStart := 0) (orderedEndExclusive := 2)
    (rhsRaw := relationRhsRaw0)
    (accumulator := LeftOperatorMerge217838.working) (working := relationWorking0)
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
end SemanticResult217842

namespace SemanticResult217845
def owner : Owner := ⟨.program ⟨257⟩, ⟨48892⟩⟩
def rawTerms : List Term := Proof.Events850.exact217845RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217845
def producerEvent : Nat := 217844
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217845.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.relationPreimageSource ⟨93⟩), 0, .finite 5647228698, .authorityRelationPreimageSource ⟨93⟩, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult217845

namespace SemanticResult217849
def owner : Owner := ⟨.program ⟨257⟩, ⟨48894⟩⟩
def rawTerms : List Term := Proof.Events850.exact217849RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217849
def producerEvent : Nat := 217848
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217849.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.scale (.predecessor 0 217846 .coefficient) (.value (.predecessor 1 217847 .coefficient)), 0, .finite 5647228698, .scale (.predecessor 0 217846 .coefficient) (.value (.predecessor 1 217847 .coefficient)), ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult217849

namespace SemanticResult217947
def owner : Owner := ⟨.program ⟨257⟩, ⟨48148⟩⟩
def rawTerms : List Term := Proof.Events851.exact217947RawTerms
def summary : Bound := .exactZero
def resultEvent : Nat := 217947
def producerEvent : Nat := 217946
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Int :=
  Cert.ResidualResult217947.actual selector witness
theorem terminalAt (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum) :
    TerminalExactAt document history (some selector) producerEvent resultEvent owner rawTerms := by
  refine ⟨by decide, ?_, ?_⟩
  · simp only [selectorMinimum] at selectorLower
    simp only [selectorMaximum] at selectorUpper
    simp [ownerAtSelector, document, owner, selectorLower, selectorUpper]
  · refine ⟨.authority (.programFamilyFact), 217910, .finite 60, .authorityProgramFamilyFact, ?_, ?_⟩
    · rfl
    · rfl
theorem claimSound (selector : Nat) (selectorLower : selectorMinimum ≤ selector)
    (selectorUpper : selector < selectorMaximum)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    ExactClaimAt history 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817 witness.env resultEvent owner
      (actual selector witness) rawTerms summary := by
  exact terminalExactClaimAt witness (terminalAt selector selectorLower selectorUpper)
end SemanticResult217947

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
