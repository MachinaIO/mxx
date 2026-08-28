import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge288606
def owner : Owner := ⟨.program ⟨257⟩, ⟨20155⟩⟩
def mergeEvent : Nat := 288606
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19673⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288602RawTerms
def rightRaw : List Term := Proof.Events1126.exact288418RawTerms
def group : MergeGroup := .operator 288602 288418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288602) (leftOrdinal := 2)
    (rightResult := 288418) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19673⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19673⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], [⟨.program ⟨257⟩, ⟨19673⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288606

namespace LeftMerge288607
def owner : Owner := ⟨.program ⟨257⟩, ⟨20155⟩⟩
def mergeEvent : Nat := 288607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288602RawTerms
def rightRaw : List Term := Proof.Events1126.exact288418RawTerms
def group : MergeGroup := .operator 288602 288418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288602) (leftOrdinal := 1)
    (rightResult := 288418) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288607

namespace LeftMerge288615
def owner : Owner := ⟨.program ⟨257⟩, ⟨20468⟩⟩
def mergeEvent : Nat := 288615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288609RawTerms
def rightRaw : List Term := Proof.Events1126.exact288334RawTerms
def group : MergeGroup := .operator 288609 288334
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288609) (leftOrdinal := 0)
    (rightResult := 288334) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20466⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288615

namespace LeftMerge288616
def owner : Owner := ⟨.program ⟨257⟩, ⟨20468⟩⟩
def mergeEvent : Nat := 288616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288609RawTerms
def rightRaw : List Term := Proof.Events1126.exact288334RawTerms
def group : MergeGroup := .operator 288609 288334
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288609) (leftOrdinal := 1)
    (rightResult := 288334) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20466⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288616

namespace LeftMerge288618
def owner : Owner := ⟨.program ⟨257⟩, ⟨20468⟩⟩
def mergeEvent : Nat := 288618
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19807⟩⟩] } }
def rhsRaw : List Term := Proof.Events1126.exact288331RawTerms
def group : MergeGroup := .relation 288617
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288617) (rhsResult := 288331)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20466⟩⟩) ⟨19807⟩ 288331) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19807⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288618

namespace LeftMerge288632
def owner : Owner := ⟨.program ⟨257⟩, ⟨19339⟩⟩
def mergeEvent : Nat := 288632
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1127.exact288626RawTerms
def group : MergeGroup := .operator 280745 288626
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 288626) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19336⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288632

namespace LeftMerge288753
def owner : Owner := ⟨.program ⟨257⟩, ⟨20044⟩⟩
def mergeEvent : Nat := 288753
def frameStart : Nat := 288687
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288749RawTerms
def rightRaw : List Term := Proof.Events1127.exact288747RawTerms
def group : MergeGroup := .operator 288749 288747
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288749) (leftOrdinal := 0)
    (rightResult := 288747) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288753

namespace LeftMerge288765
def owner : Owner := ⟨.program ⟨257⟩, ⟨20467⟩⟩
def mergeEvent : Nat := 288765
def frameStart : Nat := 288687
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288761RawTerms
def rightRaw : List Term := Proof.Events1127.exact288738RawTerms
def group : MergeGroup := .operator 288761 288738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288761) (leftOrdinal := 0)
    (rightResult := 288738) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20466⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288765

namespace LeftMerge288766
def owner : Owner := ⟨.program ⟨257⟩, ⟨20467⟩⟩
def mergeEvent : Nat := 288766
def frameStart : Nat := 288687
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288761RawTerms
def rightRaw : List Term := Proof.Events1127.exact288738RawTerms
def group : MergeGroup := .operator 288761 288738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288761) (leftOrdinal := 1)
    (rightResult := 288738) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20466⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288766

namespace LeftMerge288768
def owner : Owner := ⟨.program ⟨257⟩, ⟨20467⟩⟩
def mergeEvent : Nat := 288768
def frameStart : Nat := 288687
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19807⟩⟩] } }
def rhsRaw : List Term := Proof.Events1127.exact288735RawTerms
def group : MergeGroup := .relation 288767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288767) (rhsResult := 288735)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20466⟩⟩) ⟨19807⟩ 288735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19807⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288768

namespace LeftMerge288776
def owner : Owner := ⟨.program ⟨257⟩, ⟨18754⟩⟩
def mergeEvent : Nat := 288776
def frameStart : Nat := 288687
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288749RawTerms
def rightRaw : List Term := Proof.Events1128.exact288772RawTerms
def group : MergeGroup := .operator 288749 288772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288749) (leftOrdinal := 0)
    (rightResult := 288772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288776

namespace LeftMerge288793
def owner : Owner := ⟨.program ⟨257⟩, ⟨19339⟩⟩
def mergeEvent : Nat := 288793
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }
def rhsRaw : List Term := Proof.Events1128.exact288790RawTerms
def group : MergeGroup := .relation 288792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288792) (rhsResult := 288790)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩) (none) 288790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288793

namespace LeftMerge288794
def owner : Owner := ⟨.program ⟨257⟩, ⟨19339⟩⟩
def mergeEvent : Nat := 288794
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩] } }
def rhsRaw : List Term := Proof.Events1128.exact288790RawTerms
def group : MergeGroup := .relation 288792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288792) (rhsResult := 288790)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩) (none) 288790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288794

namespace LeftMerge288795
def owner : Owner := ⟨.program ⟨257⟩, ⟨19339⟩⟩
def mergeEvent : Nat := 288795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19807⟩⟩] } }
def rhsRaw : List Term := Proof.Events1128.exact288790RawTerms
def group : MergeGroup := .relation 288792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288792) (rhsResult := 288790)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩) (none) 288790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19807⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨19807⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288795

namespace LeftMerge288796
def owner : Owner := ⟨.program ⟨257⟩, ⟨19339⟩⟩
def mergeEvent : Nat := 288796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1128.exact288790RawTerms
def group : MergeGroup := .relation 288792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288792) (rhsResult := 288790)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19336⟩⟩]⟩) (none) 288790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18752⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18752⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288796

namespace LeftMerge288801
def owner : Owner := ⟨.program ⟨257⟩, ⟨20469⟩⟩
def mergeEvent : Nat := 288801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩] } }
def leftRaw : List Term := Proof.Events1128.exact288797RawTerms
def rightRaw : List Term := Proof.Events1127.exact288619RawTerms
def group : MergeGroup := .operator 288797 288619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288797) (leftOrdinal := 0)
    (rightResult := 288619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20466⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288801

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
