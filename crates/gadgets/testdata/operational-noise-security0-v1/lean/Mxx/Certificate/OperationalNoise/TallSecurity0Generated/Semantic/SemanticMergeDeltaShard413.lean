import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge67321
def owner : Owner := ⟨.program ⟨214⟩, ⟨19959⟩⟩
def mergeEvent : Nat := 67321
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events262.exact67315RawTerms
def group : MergeGroup := .operator 65387 67315
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 67315) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19956⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67321

namespace LeftMerge67400
def owner : Owner := ⟨.program ⟨214⟩, ⟨12559⟩⟩
def mergeEvent : Nat := 67400
def frameStart : Nat := 67370
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events263.exact67396RawTerms
def rightRaw : List Term := Proof.Events263.exact67393RawTerms
def group : MergeGroup := .operator 67396 67393
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67396) (leftOrdinal := 0)
    (rightResult := 67393) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9920⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67400

namespace LeftMerge67430
def owner : Owner := ⟨.program ⟨214⟩, ⟨12660⟩⟩
def mergeEvent : Nat := 67430
def frameStart : Nat := 67370
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67426RawTerms
def rightRaw : List Term := Proof.Events263.exact67424RawTerms
def group : MergeGroup := .operator 67426 67424
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67426) (leftOrdinal := 0)
    (rightResult := 67424) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67430

namespace LeftMerge67453
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def mergeEvent : Nat := 67453
def frameStart : Nat := 67370
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67449RawTerms
def rightRaw : List Term := Proof.Events263.exact67446RawTerms
def group : MergeGroup := .operator 67449 67446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67449) (leftOrdinal := 0)
    (rightResult := 67446) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7870⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67453

namespace LeftMerge67462
def owner : Owner := ⟨.program ⟨214⟩, ⟨25448⟩⟩
def mergeEvent : Nat := 67462
def frameStart : Nat := 67370
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67458RawTerms
def rightRaw : List Term := Proof.Events263.exact67415RawTerms
def group : MergeGroup := .operator 67458 67415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67458) (leftOrdinal := 0)
    (rightResult := 67415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25445⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67462

namespace LeftMerge67463
def owner : Owner := ⟨.program ⟨214⟩, ⟨25448⟩⟩
def mergeEvent : Nat := 67463
def frameStart : Nat := 67370
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67458RawTerms
def rightRaw : List Term := Proof.Events263.exact67415RawTerms
def group : MergeGroup := .operator 67458 67415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67458) (leftOrdinal := 1)
    (rightResult := 67415) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25445⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67463

namespace LeftMerge67465
def owner : Owner := ⟨.program ⟨214⟩, ⟨25448⟩⟩
def mergeEvent : Nat := 67465
def frameStart : Nat := 67370
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23246⟩⟩] } }
def rhsRaw : List Term := Proof.Events263.exact67412RawTerms
def group : MergeGroup := .relation 67464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67464) (rhsResult := 67412)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25445⟩⟩) ⟨23246⟩ 67412) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23246⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], [⟨.program ⟨214⟩, ⟨23246⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67465

namespace LeftMerge67473
def owner : Owner := ⟨.program ⟨214⟩, ⟨16547⟩⟩
def mergeEvent : Nat := 67473
def frameStart : Nat := 67370
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16545⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67426RawTerms
def rightRaw : List Term := Proof.Events263.exact67469RawTerms
def group : MergeGroup := .operator 67426 67469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67426) (leftOrdinal := 0)
    (rightResult := 67469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16545⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16545⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67473

namespace LeftMerge67490
def owner : Owner := ⟨.program ⟨214⟩, ⟨19959⟩⟩
def mergeEvent : Nat := 67490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }
def rhsRaw : List Term := Proof.Events263.exact67487RawTerms
def group : MergeGroup := .relation 67489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67489) (rhsResult := 67487)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67488 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩) (none) 67487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67490

namespace LeftMerge67491
def owner : Owner := ⟨.program ⟨214⟩, ⟨19959⟩⟩
def mergeEvent : Nat := 67491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩] } }
def rhsRaw : List Term := Proof.Events263.exact67487RawTerms
def group : MergeGroup := .relation 67489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67489) (rhsResult := 67487)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67488 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩) (none) 67487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67491

namespace LeftMerge67492
def owner : Owner := ⟨.program ⟨214⟩, ⟨19959⟩⟩
def mergeEvent : Nat := 67492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23246⟩⟩] } }
def rhsRaw : List Term := Proof.Events263.exact67487RawTerms
def group : MergeGroup := .relation 67489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67489) (rhsResult := 67487)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67488 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩) (none) 67487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23246⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], [⟨.program ⟨214⟩, ⟨23246⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67492

namespace LeftMerge67493
def owner : Owner := ⟨.program ⟨214⟩, ⟨19959⟩⟩
def mergeEvent : Nat := 67493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16545⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events263.exact67487RawTerms
def group : MergeGroup := .relation 67489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 67489) (rhsResult := 67487)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 67488 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩) (none) 67487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16545⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16545⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67493

namespace LeftMerge67498
def owner : Owner := ⟨.program ⟨214⟩, ⟨25447⟩⟩
def mergeEvent : Nat := 67498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23246⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67494RawTerms
def rightRaw : List Term := Proof.Events262.exact67308RawTerms
def group : MergeGroup := .operator 67494 67308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67494) (leftOrdinal := 2)
    (rightResult := 67308) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23246⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23246⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨9920⟩⟩, ⟨.program ⟨214⟩, ⟨12558⟩⟩], [⟨.program ⟨214⟩, ⟨23246⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67498

namespace LeftMerge67499
def owner : Owner := ⟨.program ⟨214⟩, ⟨25447⟩⟩
def mergeEvent : Nat := 67499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67494RawTerms
def rightRaw : List Term := Proof.Events262.exact67308RawTerms
def group : MergeGroup := .operator 67494 67308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67494) (leftOrdinal := 1)
    (rightResult := 67308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25445⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67499

namespace LeftMerge67507
def owner : Owner := ⟨.program ⟨214⟩, ⟨29157⟩⟩
def mergeEvent : Nat := 67507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29155⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67501RawTerms
def rightRaw : List Term := Proof.Events262.exact67224RawTerms
def group : MergeGroup := .operator 67501 67224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67501) (leftOrdinal := 0)
    (rightResult := 67224) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29155⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29155⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge67507

namespace LeftMerge67508
def owner : Owner := ⟨.program ⟨214⟩, ⟨29157⟩⟩
def mergeEvent : Nat := 67508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16545⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29155⟩⟩] } }
def leftRaw : List Term := Proof.Events263.exact67501RawTerms
def rightRaw : List Term := Proof.Events262.exact67224RawTerms
def group : MergeGroup := .operator 67501 67224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67501) (leftOrdinal := 1)
    (rightResult := 67224) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16545⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29155⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16545⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29155⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge67508

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
