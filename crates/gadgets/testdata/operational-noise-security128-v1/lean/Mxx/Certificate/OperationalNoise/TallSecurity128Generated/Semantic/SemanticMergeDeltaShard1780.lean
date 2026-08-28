import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge288406
def owner : Owner := ⟨.program ⟨257⟩, ⟨18137⟩⟩
def mergeEvent : Nat := 288406
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def leftRaw : List Term := Proof.Events1126.exact288402RawTerms
def rightRaw : List Term := Proof.Events1126.exact288372RawTerms
def group : MergeGroup := .operator 288402 288372
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288402) (leftOrdinal := 1)
    (rightResult := 288372) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288406

namespace LeftMerge288414
def owner : Owner := ⟨.program ⟨257⟩, ⟨20154⟩⟩
def mergeEvent : Nat := 288414
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩] } }
def leftRaw : List Term := Proof.Events1126.exact288408RawTerms
def rightRaw : List Term := Proof.Events1126.exact288344RawTerms
def group : MergeGroup := .operator 288408 288344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288408) (leftOrdinal := 1)
    (rightResult := 288344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288414

namespace LeftMerge288416
def owner : Owner := ⟨.program ⟨257⟩, ⟨20154⟩⟩
def mergeEvent : Nat := 288416
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19673⟩⟩] } }
def rhsRaw : List Term := Proof.Events1126.exact288341RawTerms
def group : MergeGroup := .relation 288415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288415) (rhsResult := 288341)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20153⟩⟩) ⟨19673⟩ 288341) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19673⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], [⟨.program ⟨257⟩, ⟨19673⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288416

namespace LeftMerge288417
def owner : Owner := ⟨.program ⟨257⟩, ⟨20154⟩⟩
def mergeEvent : Nat := 288417
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩] } }
def leftRaw : List Term := Proof.Events1126.exact288408RawTerms
def rightRaw : List Term := Proof.Events1126.exact288344RawTerms
def group : MergeGroup := .operator 288408 288344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288408) (leftOrdinal := 0)
    (rightResult := 288344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288417

namespace LeftMerge288431
def owner : Owner := ⟨.program ⟨257⟩, ⟨19092⟩⟩
def mergeEvent : Nat := 288431
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1126.exact288425RawTerms
def group : MergeGroup := .operator 280745 288425
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 288425) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19089⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288431

namespace LeftMerge288510
def owner : Owner := ⟨.program ⟨257⟩, ⟨18131⟩⟩
def mergeEvent : Nat := 288510
def frameStart : Nat := 288480
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1126.exact288506RawTerms
def rightRaw : List Term := Proof.Events1126.exact288503RawTerms
def group : MergeGroup := .operator 288506 288503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288506) (leftOrdinal := 0)
    (rightResult := 288503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12591⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288510

namespace LeftMerge288540
def owner : Owner := ⟨.program ⟨257⟩, ⟨19964⟩⟩
def mergeEvent : Nat := 288540
def frameStart : Nat := 288480
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288536RawTerms
def rightRaw : List Term := Proof.Events1127.exact288534RawTerms
def group : MergeGroup := .operator 288536 288534
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288536) (leftOrdinal := 0)
    (rightResult := 288534) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288540

namespace LeftMerge288561
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def mergeEvent : Nat := 288561
def frameStart : Nat := 288480
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288557RawTerms
def rightRaw : List Term := Proof.Events1127.exact288554RawTerms
def group : MergeGroup := .operator 288557 288554
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288557) (leftOrdinal := 0)
    (rightResult := 288554) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288561

namespace LeftMerge288570
def owner : Owner := ⟨.program ⟨257⟩, ⟨20156⟩⟩
def mergeEvent : Nat := 288570
def frameStart : Nat := 288480
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288566RawTerms
def rightRaw : List Term := Proof.Events1127.exact288525RawTerms
def group : MergeGroup := .operator 288566 288525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288566) (leftOrdinal := 0)
    (rightResult := 288525) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20153⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288570

namespace LeftMerge288571
def owner : Owner := ⟨.program ⟨257⟩, ⟨20156⟩⟩
def mergeEvent : Nat := 288571
def frameStart : Nat := 288480
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288566RawTerms
def rightRaw : List Term := Proof.Events1127.exact288525RawTerms
def group : MergeGroup := .operator 288566 288525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288566) (leftOrdinal := 1)
    (rightResult := 288525) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288571

namespace LeftMerge288573
def owner : Owner := ⟨.program ⟨257⟩, ⟨20156⟩⟩
def mergeEvent : Nat := 288573
def frameStart : Nat := 288480
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19673⟩⟩] } }
def rhsRaw : List Term := Proof.Events1127.exact288522RawTerms
def group : MergeGroup := .relation 288572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288572) (rhsResult := 288522)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20153⟩⟩) ⟨19673⟩ 288522) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19673⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], [⟨.program ⟨257⟩, ⟨19673⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288573

namespace LeftMerge288581
def owner : Owner := ⟨.program ⟨257⟩, ⟨18542⟩⟩
def mergeEvent : Nat := 288581
def frameStart : Nat := 288480
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1127.exact288536RawTerms
def rightRaw : List Term := Proof.Events1127.exact288577RawTerms
def group : MergeGroup := .operator 288536 288577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 288536) (leftOrdinal := 0)
    (rightResult := 288577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288581

namespace LeftMerge288598
def owner : Owner := ⟨.program ⟨257⟩, ⟨19092⟩⟩
def mergeEvent : Nat := 288598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events1127.exact288595RawTerms
def group : MergeGroup := .relation 288597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288597) (rhsResult := 288595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288596 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩]⟩) (none) 288595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288598

namespace LeftMerge288599
def owner : Owner := ⟨.program ⟨257⟩, ⟨19092⟩⟩
def mergeEvent : Nat := 288599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩] } }
def rhsRaw : List Term := Proof.Events1127.exact288595RawTerms
def group : MergeGroup := .relation 288597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288597) (rhsResult := 288595)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288596 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩]⟩) (none) 288595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288599

namespace LeftMerge288600
def owner : Owner := ⟨.program ⟨257⟩, ⟨19092⟩⟩
def mergeEvent : Nat := 288600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19673⟩⟩] } }
def rhsRaw : List Term := Proof.Events1127.exact288595RawTerms
def group : MergeGroup := .relation 288597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288597) (rhsResult := 288595)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288596 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩]⟩) (none) 288595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19673⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨12591⟩⟩, ⟨.program ⟨257⟩, ⟨18130⟩⟩], [⟨.program ⟨257⟩, ⟨19673⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge288600

namespace LeftMerge288601
def owner : Owner := ⟨.program ⟨257⟩, ⟨19092⟩⟩
def mergeEvent : Nat := 288601
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1127.exact288595RawTerms
def group : MergeGroup := .relation 288597
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 288597) (rhsResult := 288595)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 288596 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19089⟩⟩]⟩) (none) 288595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨18540⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge288601

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
