import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge173769
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173769
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 13)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173769

namespace LeftMerge173770
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173770
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 12)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173770

namespace LeftMerge173771
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173771
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 11)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173771

namespace LeftMerge173772
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173772
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 10)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173772

namespace LeftMerge173773
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173773
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 9)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173773

namespace LeftMerge173774
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173774
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 8)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173774

namespace LeftMerge173775
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173775
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 7)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173775

namespace LeftMerge173776
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173776
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 6)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173776

namespace LeftMerge173777
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173777
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 5)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173777

namespace LeftMerge173778
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173778
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 4)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173778

namespace LeftMerge173779
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173779
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 3)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173779

namespace LeftMerge173780
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173780
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 2)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173780

namespace LeftMerge173781
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173781
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 1)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173781

namespace LeftMerge173782
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173782
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 0)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge173782

namespace LeftMerge173783
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173783
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48415⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩] } }
def leftRaw : List Term := Proof.Events678.exact173761RawTerms
def rightRaw : List Term := Proof.Events678.exact173602RawTerms
def group : MergeGroup := .operator 173761 173602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 173761) (leftOrdinal := 29)
    (rightResult := 173602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48415⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173783

namespace LeftMerge173785
def owner : Owner := ⟨.program ⟨257⟩, ⟨71366⟩⟩
def mergeEvent : Nat := 173785
def frameStart : Nat := 173086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48415⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }
def rhsRaw : List Term := Proof.Events678.exact173599RawTerms
def group : MergeGroup := .relation 173784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 173784) (rhsResult := 173599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71365⟩⟩) ⟨68854⟩ 173599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68854⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨68854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge173785

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
