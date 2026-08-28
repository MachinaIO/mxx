import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge51622
def owner : Owner := ⟨.program ⟨214⟩, ⟨29835⟩⟩
def mergeEvent : Nat := 51622
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29832⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51618RawTerms
def rightRaw : List Term := Proof.Events200.exact51440RawTerms
def group : MergeGroup := .operator 51618 51440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51618) (leftOrdinal := 0)
    (rightResult := 51440) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29832⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29832⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29832⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51622

namespace LeftMerge51623
def owner : Owner := ⟨.program ⟨214⟩, ⟨29835⟩⟩
def mergeEvent : Nat := 51623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16875⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24732⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51618RawTerms
def rightRaw : List Term := Proof.Events200.exact51440RawTerms
def group : MergeGroup := .operator 51618 51440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51618) (leftOrdinal := 2)
    (rightResult := 51440) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16875⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24732⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16875⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24732⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16875⟩⟩], [⟨.program ⟨214⟩, ⟨24732⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51623

namespace LeftMerge51649
def owner : Owner := ⟨.program ⟨214⟩, ⟨12969⟩⟩
def mergeEvent : Nat := 51649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2384RawTerms
def rightRaw : List Term := Proof.Events197.exact50670RawTerms
def group : MergeGroup := .operator 2384 50670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2384) (leftOrdinal := 0)
    (rightResult := 50670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51649

namespace LeftMerge51654
def owner : Owner := ⟨.program ⟨214⟩, ⟨7282⟩⟩
def mergeEvent : Nat := 51654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50540RawTerms
def rightRaw : List Term := Proof.Events029.exact7474RawTerms
def group : MergeGroup := .operator 50540 7474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50540) (leftOrdinal := 0)
    (rightResult := 7474) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51654

namespace LeftMerge51671
def owner : Owner := ⟨.program ⟨214⟩, ⟨12972⟩⟩
def mergeEvent : Nat := 51671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51665RawTerms
def rightRaw : List Term := Proof.Events009.exact2387RawTerms
def group : MergeGroup := .operator 51665 2387
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51665) (leftOrdinal := 1)
    (rightResult := 2387) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51671

namespace LeftMerge51672
def owner : Owner := ⟨.program ⟨214⟩, ⟨12972⟩⟩
def mergeEvent : Nat := 51672
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51665RawTerms
def rightRaw : List Term := Proof.Events009.exact2387RawTerms
def group : MergeGroup := .operator 51665 2387
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51665) (leftOrdinal := 0)
    (rightResult := 2387) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51672

namespace LeftMerge51677
def owner : Owner := ⟨.program ⟨214⟩, ⟨10141⟩⟩
def mergeEvent : Nat := 51677
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2387RawTerms
def rightRaw : List Term := Proof.Events197.exact50670RawTerms
def group : MergeGroup := .operator 2387 50670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2387) (leftOrdinal := 0)
    (rightResult := 50670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51677

namespace LeftMerge51682
def owner : Owner := ⟨.program ⟨214⟩, ⟨7262⟩⟩
def mergeEvent : Nat := 51682
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }
def leftRaw : List Term := Proof.Events197.exact50540RawTerms
def rightRaw : List Term := Proof.Events029.exact7515RawTerms
def group : MergeGroup := .operator 50540 7515
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50540) (leftOrdinal := 0)
    (rightResult := 7515) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51682

namespace LeftMerge51699
def owner : Owner := ⟨.program ⟨214⟩, ⟨10144⟩⟩
def mergeEvent : Nat := 51699
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51693RawTerms
def rightRaw : List Term := Proof.Events029.exact7504RawTerms
def group : MergeGroup := .operator 51693 7504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51693) (leftOrdinal := 1)
    (rightResult := 7504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51699

namespace LeftMerge51701
def owner : Owner := ⟨.program ⟨214⟩, ⟨10144⟩⟩
def mergeEvent : Nat := 51701
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def rhsRaw : List Term := Proof.Events029.exact7474RawTerms
def group : MergeGroup := .relation 51700
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51700) (rhsResult := 7474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7876⟩⟩) ⟨6788⟩ 7474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51701

namespace LeftMerge51702
def owner : Owner := ⟨.program ⟨214⟩, ⟨10144⟩⟩
def mergeEvent : Nat := 51702
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51693RawTerms
def rightRaw : List Term := Proof.Events029.exact7504RawTerms
def group : MergeGroup := .operator 51693 7504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51693) (leftOrdinal := 0)
    (rightResult := 7504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51702

namespace LeftMerge51707
def owner : Owner := ⟨.program ⟨214⟩, ⟨12973⟩⟩
def mergeEvent : Nat := 51707
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51703RawTerms
def rightRaw : List Term := Proof.Events201.exact51673RawTerms
def group : MergeGroup := .operator 51703 51673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51703) (leftOrdinal := 1)
    (rightResult := 51673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51707

namespace LeftMerge51715
def owner : Owner := ⟨.program ⟨214⟩, ⟨25610⟩⟩
def mergeEvent : Nat := 51715
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51709RawTerms
def rightRaw : List Term := Proof.Events201.exact51645RawTerms
def group : MergeGroup := .operator 51709 51645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51709) (leftOrdinal := 1)
    (rightResult := 51645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25609⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51715

namespace LeftMerge51717
def owner : Owner := ⟨.program ⟨214⟩, ⟨25610⟩⟩
def mergeEvent : Nat := 51717
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23334⟩⟩] } }
def rhsRaw : List Term := Proof.Events201.exact51642RawTerms
def group : MergeGroup := .relation 51716
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51716) (rhsResult := 51642)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25609⟩⟩) ⟨23334⟩ 51642) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23334⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨23334⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51717

namespace LeftMerge51718
def owner : Owner := ⟨.program ⟨214⟩, ⟨25610⟩⟩
def mergeEvent : Nat := 51718
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51709RawTerms
def rightRaw : List Term := Proof.Events201.exact51645RawTerms
def group : MergeGroup := .operator 51709 51645
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51709) (leftOrdinal := 0)
    (rightResult := 51645) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25609⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51718

namespace LeftMerge51732
def owner : Owner := ⟨.program ⟨214⟩, ⟨20111⟩⟩
def mergeEvent : Nat := 51732
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events202.exact51726RawTerms
def group : MergeGroup := .operator 50762 51726
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 51726) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20108⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51732

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
