import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge257270
def owner : Owner := ⟨.program ⟨257⟩, ⟨55445⟩⟩
def mergeEvent : Nat := 257270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54959⟩⟩] } }
def rhsRaw : List Term := Proof.Events1004.exact257195RawTerms
def group : MergeGroup := .relation 257269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257269) (rhsResult := 257195)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55444⟩⟩) ⟨54959⟩ 257195) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], [⟨.program ⟨257⟩, ⟨54959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257270

namespace LeftMerge257271
def owner : Owner := ⟨.program ⟨257⟩, ⟨55445⟩⟩
def mergeEvent : Nat := 257271
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩] } }
def leftRaw : List Term := Proof.Events1004.exact257262RawTerms
def rightRaw : List Term := Proof.Events1004.exact257198RawTerms
def group : MergeGroup := .operator 257262 257198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257262) (leftOrdinal := 0)
    (rightResult := 257198) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55444⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257271

namespace LeftMerge257285
def owner : Owner := ⟨.program ⟨257⟩, ⟨54382⟩⟩
def mergeEvent : Nat := 257285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1004.exact257279RawTerms
def group : MergeGroup := .operator 251495 257279
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 257279) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257285

namespace LeftMerge257364
def owner : Owner := ⟨.program ⟨257⟩, ⟨53391⟩⟩
def mergeEvent : Nat := 257364
def frameStart : Nat := 257334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1005.exact257360RawTerms
def rightRaw : List Term := Proof.Events1005.exact257357RawTerms
def group : MergeGroup := .operator 257360 257357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257360) (leftOrdinal := 0)
    (rightResult := 257357) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24710⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257364

namespace LeftMerge257394
def owner : Owner := ⟨.program ⟨257⟩, ⟨55248⟩⟩
def mergeEvent : Nat := 257394
def frameStart : Nat := 257334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1005.exact257390RawTerms
def rightRaw : List Term := Proof.Events1005.exact257388RawTerms
def group : MergeGroup := .operator 257390 257388
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257390) (leftOrdinal := 0)
    (rightResult := 257388) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257394

namespace LeftMerge257417
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 257417
def frameStart : Nat := 257334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events1005.exact257413RawTerms
def rightRaw : List Term := Proof.Events1005.exact257410RawTerms
def group : MergeGroup := .operator 257413 257410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257413) (leftOrdinal := 0)
    (rightResult := 257410) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257417

namespace LeftMerge257426
def owner : Owner := ⟨.program ⟨257⟩, ⟨55447⟩⟩
def mergeEvent : Nat := 257426
def frameStart : Nat := 257334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩] } }
def leftRaw : List Term := Proof.Events1005.exact257422RawTerms
def rightRaw : List Term := Proof.Events1005.exact257379RawTerms
def group : MergeGroup := .operator 257422 257379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257422) (leftOrdinal := 0)
    (rightResult := 257379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55444⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257426

namespace LeftMerge257427
def owner : Owner := ⟨.program ⟨257⟩, ⟨55447⟩⟩
def mergeEvent : Nat := 257427
def frameStart : Nat := 257334
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩] } }
def leftRaw : List Term := Proof.Events1005.exact257422RawTerms
def rightRaw : List Term := Proof.Events1005.exact257379RawTerms
def group : MergeGroup := .operator 257422 257379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257422) (leftOrdinal := 1)
    (rightResult := 257379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55444⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257427

namespace LeftMerge257429
def owner : Owner := ⟨.program ⟨257⟩, ⟨55447⟩⟩
def mergeEvent : Nat := 257429
def frameStart : Nat := 257334
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54959⟩⟩] } }
def rhsRaw : List Term := Proof.Events1005.exact257376RawTerms
def group : MergeGroup := .relation 257428
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257428) (rhsResult := 257376)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55444⟩⟩) ⟨54959⟩ 257376) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], [⟨.program ⟨257⟩, ⟨54959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257429

namespace LeftMerge257437
def owner : Owner := ⟨.program ⟨257⟩, ⟨53830⟩⟩
def mergeEvent : Nat := 257437
def frameStart : Nat := 257334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1005.exact257390RawTerms
def rightRaw : List Term := Proof.Events1005.exact257433RawTerms
def group : MergeGroup := .operator 257390 257433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257390) (leftOrdinal := 0)
    (rightResult := 257433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257437

namespace LeftMerge257454
def owner : Owner := ⟨.program ⟨257⟩, ⟨54382⟩⟩
def mergeEvent : Nat := 257454
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events1005.exact257451RawTerms
def group : MergeGroup := .relation 257453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257453) (rhsResult := 257451)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 257452 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩]⟩) (none) 257451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257454

namespace LeftMerge257455
def owner : Owner := ⟨.program ⟨257⟩, ⟨54382⟩⟩
def mergeEvent : Nat := 257455
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩] } }
def rhsRaw : List Term := Proof.Events1005.exact257451RawTerms
def group : MergeGroup := .relation 257453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257453) (rhsResult := 257451)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 257452 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩]⟩) (none) 257451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257455

namespace LeftMerge257456
def owner : Owner := ⟨.program ⟨257⟩, ⟨54382⟩⟩
def mergeEvent : Nat := 257456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54959⟩⟩] } }
def rhsRaw : List Term := Proof.Events1005.exact257451RawTerms
def group : MergeGroup := .relation 257453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257453) (rhsResult := 257451)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 257452 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩]⟩) (none) 257451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], [⟨.program ⟨257⟩, ⟨54959⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257456

namespace LeftMerge257457
def owner : Owner := ⟨.program ⟨257⟩, ⟨54382⟩⟩
def mergeEvent : Nat := 257457
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨53828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1005.exact257451RawTerms
def group : MergeGroup := .relation 257453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257453) (rhsResult := 257451)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 257452 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54379⟩⟩]⟩) (none) 257451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨53828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257457

namespace LeftMerge257462
def owner : Owner := ⟨.program ⟨257⟩, ⟨55446⟩⟩
def mergeEvent : Nat := 257462
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54959⟩⟩] } }
def leftRaw : List Term := Proof.Events1005.exact257458RawTerms
def rightRaw : List Term := Proof.Events1004.exact257272RawTerms
def group : MergeGroup := .operator 257458 257272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257458) (leftOrdinal := 2)
    (rightResult := 257272) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54959⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54959⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24710⟩⟩, ⟨.program ⟨257⟩, ⟨53390⟩⟩], [⟨.program ⟨257⟩, ⟨54959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257462

namespace LeftMerge257463
def owner : Owner := ⟨.program ⟨257⟩, ⟨55446⟩⟩
def mergeEvent : Nat := 257463
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩] } }
def leftRaw : List Term := Proof.Events1005.exact257458RawTerms
def rightRaw : List Term := Proof.Events1004.exact257272RawTerms
def group : MergeGroup := .operator 257458 257272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257458) (leftOrdinal := 1)
    (rightResult := 257272) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55444⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257463

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
