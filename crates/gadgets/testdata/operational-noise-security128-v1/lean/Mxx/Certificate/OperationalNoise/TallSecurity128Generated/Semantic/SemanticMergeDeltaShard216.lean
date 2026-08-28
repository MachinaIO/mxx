import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge39273
def owner : Owner := ⟨.program ⟨257⟩, ⟨21713⟩⟩
def mergeEvent : Nat := 39273
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events004.exact1187RawTerms
def rightRaw : List Term := Proof.Events125.exact32028RawTerms
def group : MergeGroup := .operator 1187 32028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1187) (leftOrdinal := 0)
    (rightResult := 32028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39273

namespace LeftMerge39278
def owner : Owner := ⟨.program ⟨257⟩, ⟨11639⟩⟩
def mergeEvent : Nat := 39278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events124.exact31898RawTerms
def rightRaw : List Term := Proof.Events096.exact24595RawTerms
def group : MergeGroup := .operator 31898 24595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31898) (leftOrdinal := 0)
    (rightResult := 24595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39278

namespace LeftMerge39295
def owner : Owner := ⟨.program ⟨257⟩, ⟨21716⟩⟩
def mergeEvent : Nat := 39295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39289RawTerms
def rightRaw : List Term := Proof.Events004.exact1190RawTerms
def group : MergeGroup := .operator 39289 1190
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39289) (leftOrdinal := 1)
    (rightResult := 1190) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39295

namespace LeftMerge39296
def owner : Owner := ⟨.program ⟨257⟩, ⟨21716⟩⟩
def mergeEvent : Nat := 39296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39289RawTerms
def rightRaw : List Term := Proof.Events004.exact1190RawTerms
def group : MergeGroup := .operator 39289 1190
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39289) (leftOrdinal := 0)
    (rightResult := 1190) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39296

namespace LeftMerge39301
def owner : Owner := ⟨.program ⟨257⟩, ⟨21237⟩⟩
def mergeEvent : Nat := 39301
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events004.exact1190RawTerms
def rightRaw : List Term := Proof.Events125.exact32028RawTerms
def group : MergeGroup := .operator 1190 32028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1190) (leftOrdinal := 0)
    (rightResult := 32028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39301

namespace LeftMerge39306
def owner : Owner := ⟨.program ⟨257⟩, ⟨11619⟩⟩
def mergeEvent : Nat := 39306
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }
def leftRaw : List Term := Proof.Events124.exact31898RawTerms
def rightRaw : List Term := Proof.Events096.exact24636RawTerms
def group : MergeGroup := .operator 31898 24636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31898) (leftOrdinal := 0)
    (rightResult := 24636) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39306

namespace LeftMerge39323
def owner : Owner := ⟨.program ⟨257⟩, ⟨21240⟩⟩
def mergeEvent : Nat := 39323
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39317RawTerms
def rightRaw : List Term := Proof.Events096.exact24625RawTerms
def group : MergeGroup := .operator 39317 24625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39317) (leftOrdinal := 1)
    (rightResult := 24625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39323

namespace LeftMerge39325
def owner : Owner := ⟨.program ⟨257⟩, ⟨21240⟩⟩
def mergeEvent : Nat := 39325
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24595RawTerms
def group : MergeGroup := .relation 39324
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39324) (rhsResult := 24595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9574⟩⟩) ⟨7306⟩ 24595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39325

namespace LeftMerge39326
def owner : Owner := ⟨.program ⟨257⟩, ⟨21240⟩⟩
def mergeEvent : Nat := 39326
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39317RawTerms
def rightRaw : List Term := Proof.Events096.exact24625RawTerms
def group : MergeGroup := .operator 39317 24625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39317) (leftOrdinal := 0)
    (rightResult := 24625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39326

namespace LeftMerge39331
def owner : Owner := ⟨.program ⟨257⟩, ⟨21717⟩⟩
def mergeEvent : Nat := 39331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39327RawTerms
def rightRaw : List Term := Proof.Events153.exact39297RawTerms
def group : MergeGroup := .operator 39327 39297
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39327) (leftOrdinal := 1)
    (rightResult := 39297) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39331

namespace LeftMerge39339
def owner : Owner := ⟨.program ⟨257⟩, ⟨23539⟩⟩
def mergeEvent : Nat := 39339
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23538⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39333RawTerms
def rightRaw : List Term := Proof.Events153.exact39269RawTerms
def group : MergeGroup := .operator 39333 39269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39333) (leftOrdinal := 1)
    (rightResult := 39269) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23538⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39339

namespace LeftMerge39341
def owner : Owner := ⟨.program ⟨257⟩, ⟨23539⟩⟩
def mergeEvent : Nat := 39341
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22983⟩⟩] } }
def rhsRaw : List Term := Proof.Events153.exact39266RawTerms
def group : MergeGroup := .relation 39340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 39340) (rhsResult := 39266)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23538⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23538⟩⟩) ⟨22983⟩ 39266) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22983⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], [⟨.program ⟨257⟩, ⟨22983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge39341

namespace LeftMerge39342
def owner : Owner := ⟨.program ⟨257⟩, ⟨23539⟩⟩
def mergeEvent : Nat := 39342
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23538⟩⟩] } }
def leftRaw : List Term := Proof.Events153.exact39333RawTerms
def rightRaw : List Term := Proof.Events153.exact39269RawTerms
def group : MergeGroup := .operator 39333 39269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39333) (leftOrdinal := 0)
    (rightResult := 39269) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39342

namespace LeftMerge39356
def owner : Owner := ⟨.program ⟨257⟩, ⟨22462⟩⟩
def mergeEvent : Nat := 39356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22459⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events153.exact39350RawTerms
def group : MergeGroup := .operator 32120 39350
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 39350) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39356

namespace LeftMerge39435
def owner : Owner := ⟨.program ⟨257⟩, ⟨21711⟩⟩
def mergeEvent : Nat := 39435
def frameStart : Nat := 39405
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events154.exact39431RawTerms
def rightRaw : List Term := Proof.Events154.exact39428RawTerms
def group : MergeGroup := .operator 39431 39428
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39431) (leftOrdinal := 0)
    (rightResult := 39428) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21236⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39435

namespace LeftMerge39465
def owner : Owner := ⟨.program ⟨257⟩, ⟨23244⟩⟩
def mergeEvent : Nat := 39465
def frameStart : Nat := 39405
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events154.exact39461RawTerms
def rightRaw : List Term := Proof.Events154.exact39459RawTerms
def group : MergeGroup := .operator 39461 39459
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39461) (leftOrdinal := 0)
    (rightResult := 39459) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21236⟩⟩, ⟨.program ⟨257⟩, ⟨21710⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge39465

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
