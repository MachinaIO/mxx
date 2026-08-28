import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge26308
def owner : Owner := ⟨.program ⟨214⟩, ⟨14241⟩⟩
def mergeEvent : Nat := 26308
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }
def leftRaw : List Term := Proof.Events102.exact26299RawTerms
def rightRaw : List Term := Proof.Events044.exact11512RawTerms
def group : MergeGroup := .operator 26299 11512
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26299) (leftOrdinal := 0)
    (rightResult := 11512) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7852⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26308

namespace LeftMerge26313
def owner : Owner := ⟨.program ⟨214⟩, ⟨14242⟩⟩
def mergeEvent : Nat := 26313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }
def leftRaw : List Term := Proof.Events102.exact26309RawTerms
def rightRaw : List Term := Proof.Events102.exact26279RawTerms
def group : MergeGroup := .operator 26309 26279
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26309) (leftOrdinal := 1)
    (rightResult := 26279) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6779⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6779⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26313

namespace LeftMerge26321
def owner : Owner := ⟨.program ⟨214⟩, ⟨26082⟩⟩
def mergeEvent : Nat := 26321
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩] } }
def leftRaw : List Term := Proof.Events102.exact26315RawTerms
def rightRaw : List Term := Proof.Events102.exact26251RawTerms
def group : MergeGroup := .operator 26315 26251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26315) (leftOrdinal := 1)
    (rightResult := 26251) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26081⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26321

namespace LeftMerge26323
def owner : Owner := ⟨.program ⟨214⟩, ⟨26082⟩⟩
def mergeEvent : Nat := 26323
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23590⟩⟩] } }
def rhsRaw : List Term := Proof.Events102.exact26248RawTerms
def group : MergeGroup := .relation 26322
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26322) (rhsResult := 26248)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26081⟩⟩) ⟨23590⟩ 26248) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23590⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26323

namespace LeftMerge26324
def owner : Owner := ⟨.program ⟨214⟩, ⟨26082⟩⟩
def mergeEvent : Nat := 26324
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩] } }
def leftRaw : List Term := Proof.Events102.exact26315RawTerms
def rightRaw : List Term := Proof.Events102.exact26251RawTerms
def group : MergeGroup := .operator 26315 26251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26315) (leftOrdinal := 0)
    (rightResult := 26251) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26081⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26324

namespace LeftMerge26338
def owner : Owner := ⟨.program ⟨214⟩, ⟨19543⟩⟩
def mergeEvent : Nat := 26338
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events102.exact26332RawTerms
def group : MergeGroup := .operator 21512 26332
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 26332) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19540⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26338

namespace LeftMerge26417
def owner : Owner := ⟨.program ⟨214⟩, ⟨14235⟩⟩
def mergeEvent : Nat := 26417
def frameStart : Nat := 26387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events103.exact26413RawTerms
def rightRaw : List Term := Proof.Events103.exact26410RawTerms
def group : MergeGroup := .operator 26413 26410
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26413) (leftOrdinal := 0)
    (rightResult := 26410) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26417

namespace LeftMerge26447
def owner : Owner := ⟨.program ⟨214⟩, ⟨14328⟩⟩
def mergeEvent : Nat := 26447
def frameStart : Nat := 26387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events103.exact26443RawTerms
def rightRaw : List Term := Proof.Events103.exact26441RawTerms
def group : MergeGroup := .operator 26443 26441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26443) (leftOrdinal := 0)
    (rightResult := 26441) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26447

namespace LeftMerge26470
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def mergeEvent : Nat := 26470
def frameStart : Nat := 26387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }
def leftRaw : List Term := Proof.Events103.exact26466RawTerms
def rightRaw : List Term := Proof.Events103.exact26463RawTerms
def group : MergeGroup := .operator 26466 26463
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26466) (leftOrdinal := 0)
    (rightResult := 26463) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7852⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26470

namespace LeftMerge26479
def owner : Owner := ⟨.program ⟨214⟩, ⟨26084⟩⟩
def mergeEvent : Nat := 26479
def frameStart : Nat := 26387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩] } }
def leftRaw : List Term := Proof.Events103.exact26475RawTerms
def rightRaw : List Term := Proof.Events103.exact26432RawTerms
def group : MergeGroup := .operator 26475 26432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26475) (leftOrdinal := 0)
    (rightResult := 26432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26081⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26479

namespace LeftMerge26480
def owner : Owner := ⟨.program ⟨214⟩, ⟨26084⟩⟩
def mergeEvent : Nat := 26480
def frameStart : Nat := 26387
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩] } }
def leftRaw : List Term := Proof.Events103.exact26475RawTerms
def rightRaw : List Term := Proof.Events103.exact26432RawTerms
def group : MergeGroup := .operator 26475 26432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26475) (leftOrdinal := 1)
    (rightResult := 26432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26081⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26480

namespace LeftMerge26482
def owner : Owner := ⟨.program ⟨214⟩, ⟨26084⟩⟩
def mergeEvent : Nat := 26482
def frameStart : Nat := 26387
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23590⟩⟩] } }
def rhsRaw : List Term := Proof.Events103.exact26429RawTerms
def group : MergeGroup := .relation 26481
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26481) (rhsResult := 26429)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26081⟩⟩) ⟨23590⟩ 26429) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23590⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26482

namespace LeftMerge26490
def owner : Owner := ⟨.program ⟨214⟩, ⟨15954⟩⟩
def mergeEvent : Nat := 26490
def frameStart : Nat := 26387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15952⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events103.exact26443RawTerms
def rightRaw : List Term := Proof.Events103.exact26486RawTerms
def group : MergeGroup := .operator 26443 26486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26443) (leftOrdinal := 0)
    (rightResult := 26486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15952⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15952⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26490

namespace LeftMerge26507
def owner : Owner := ⟨.program ⟨214⟩, ⟨19543⟩⟩
def mergeEvent : Nat := 26507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }
def rhsRaw : List Term := Proof.Events103.exact26504RawTerms
def group : MergeGroup := .relation 26506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26506) (rhsResult := 26504)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 26505 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩) (none) 26504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6697⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6697⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26507

namespace LeftMerge26508
def owner : Owner := ⟨.program ⟨214⟩, ⟨19543⟩⟩
def mergeEvent : Nat := 26508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩] } }
def rhsRaw : List Term := Proof.Events103.exact26504RawTerms
def group : MergeGroup := .relation 26506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26506) (rhsResult := 26504)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 26505 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩) (none) 26504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6759⟩⟩, ⟨.program ⟨214⟩, ⟨7852⟩⟩, ⟨.program ⟨214⟩, ⟨26081⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26508

namespace LeftMerge26509
def owner : Owner := ⟨.program ⟨214⟩, ⟨19543⟩⟩
def mergeEvent : Nat := 26509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23590⟩⟩] } }
def rhsRaw : List Term := Proof.Events103.exact26504RawTerms
def group : MergeGroup := .relation 26506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26506) (rhsResult := 26504)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 26505 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19540⟩⟩]⟩) (none) 26504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23590⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11481⟩⟩, ⟨.program ⟨214⟩, ⟨14234⟩⟩], [⟨.program ⟨214⟩, ⟨23590⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26509

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
