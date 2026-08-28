import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge127558
def owner : Owner := ⟨.program ⟨257⟩, ⟨12625⟩⟩
def mergeEvent : Nat := 127558
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events498.exact127549RawTerms
def rightRaw : List Term := Proof.Events098.exact25126RawTerms
def group : MergeGroup := .operator 127549 25126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127549) (leftOrdinal := 0)
    (rightResult := 25126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127558

namespace LeftMerge127563
def owner : Owner := ⟨.program ⟨257⟩, ⟨18185⟩⟩
def mergeEvent : Nat := 127563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def leftRaw : List Term := Proof.Events498.exact127559RawTerms
def rightRaw : List Term := Proof.Events498.exact127529RawTerms
def group : MergeGroup := .operator 127559 127529
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127559) (leftOrdinal := 1)
    (rightResult := 127529) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127563

namespace LeftMerge127571
def owner : Owner := ⟨.program ⟨257⟩, ⟨20176⟩⟩
def mergeEvent : Nat := 127571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩] } }
def leftRaw : List Term := Proof.Events498.exact127565RawTerms
def rightRaw : List Term := Proof.Events498.exact127501RawTerms
def group : MergeGroup := .operator 127565 127501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127565) (leftOrdinal := 1)
    (rightResult := 127501) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20175⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127571

namespace LeftMerge127573
def owner : Owner := ⟨.program ⟨257⟩, ⟨20176⟩⟩
def mergeEvent : Nat := 127573
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19685⟩⟩] } }
def rhsRaw : List Term := Proof.Events498.exact127498RawTerms
def group : MergeGroup := .relation 127572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127572) (rhsResult := 127498)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20175⟩⟩) ⟨19685⟩ 127498) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19685⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], [⟨.program ⟨257⟩, ⟨19685⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127573

namespace LeftMerge127574
def owner : Owner := ⟨.program ⟨257⟩, ⟨20176⟩⟩
def mergeEvent : Nat := 127574
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩] } }
def leftRaw : List Term := Proof.Events498.exact127565RawTerms
def rightRaw : List Term := Proof.Events498.exact127501RawTerms
def group : MergeGroup := .operator 127565 127501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127565) (leftOrdinal := 0)
    (rightResult := 127501) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20175⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127574

namespace LeftMerge127588
def owner : Owner := ⟨.program ⟨257⟩, ⟨19112⟩⟩
def mergeEvent : Nat := 127588
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events498.exact127582RawTerms
def group : MergeGroup := .operator 119870 127582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 127582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19109⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127588

namespace LeftMerge127667
def owner : Owner := ⟨.program ⟨257⟩, ⟨18179⟩⟩
def mergeEvent : Nat := 127667
def frameStart : Nat := 127637
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events498.exact127663RawTerms
def rightRaw : List Term := Proof.Events498.exact127660RawTerms
def group : MergeGroup := .operator 127663 127660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127663) (leftOrdinal := 0)
    (rightResult := 127660) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12621⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127667

namespace LeftMerge127697
def owner : Owner := ⟨.program ⟨257⟩, ⟨19972⟩⟩
def mergeEvent : Nat := 127697
def frameStart : Nat := 127637
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events498.exact127693RawTerms
def rightRaw : List Term := Proof.Events498.exact127691RawTerms
def group : MergeGroup := .operator 127693 127691
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127693) (leftOrdinal := 0)
    (rightResult := 127691) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127697

namespace LeftMerge127720
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def mergeEvent : Nat := 127720
def frameStart : Nat := 127637
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events498.exact127716RawTerms
def rightRaw : List Term := Proof.Events498.exact127713RawTerms
def group : MergeGroup := .operator 127716 127713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127716) (leftOrdinal := 0)
    (rightResult := 127713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127720

namespace LeftMerge127729
def owner : Owner := ⟨.program ⟨257⟩, ⟨20178⟩⟩
def mergeEvent : Nat := 127729
def frameStart : Nat := 127637
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩] } }
def leftRaw : List Term := Proof.Events498.exact127725RawTerms
def rightRaw : List Term := Proof.Events498.exact127682RawTerms
def group : MergeGroup := .operator 127725 127682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127725) (leftOrdinal := 0)
    (rightResult := 127682) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20175⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127729

namespace LeftMerge127730
def owner : Owner := ⟨.program ⟨257⟩, ⟨20178⟩⟩
def mergeEvent : Nat := 127730
def frameStart : Nat := 127637
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩] } }
def leftRaw : List Term := Proof.Events498.exact127725RawTerms
def rightRaw : List Term := Proof.Events498.exact127682RawTerms
def group : MergeGroup := .operator 127725 127682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127725) (leftOrdinal := 1)
    (rightResult := 127682) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20175⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127730

namespace LeftMerge127732
def owner : Owner := ⟨.program ⟨257⟩, ⟨20178⟩⟩
def mergeEvent : Nat := 127732
def frameStart : Nat := 127637
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19685⟩⟩] } }
def rhsRaw : List Term := Proof.Events498.exact127679RawTerms
def group : MergeGroup := .relation 127731
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127731) (rhsResult := 127679)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20175⟩⟩) ⟨19685⟩ 127679) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19685⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], [⟨.program ⟨257⟩, ⟨19685⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127732

namespace LeftMerge127740
def owner : Owner := ⟨.program ⟨257⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 127740
def frameStart : Nat := 127637
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events498.exact127693RawTerms
def rightRaw : List Term := Proof.Events498.exact127736RawTerms
def group : MergeGroup := .operator 127693 127736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127693) (leftOrdinal := 0)
    (rightResult := 127736) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127740

namespace LeftMerge127757
def owner : Owner := ⟨.program ⟨257⟩, ⟨19112⟩⟩
def mergeEvent : Nat := 127757
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127754RawTerms
def group : MergeGroup := .relation 127756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127756) (rhsResult := 127754)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127755 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩]⟩) (none) 127754) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127757

namespace LeftMerge127758
def owner : Owner := ⟨.program ⟨257⟩, ⟨19112⟩⟩
def mergeEvent : Nat := 127758
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127754RawTerms
def group : MergeGroup := .relation 127756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127756) (rhsResult := 127754)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127755 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩]⟩) (none) 127754) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127758

namespace LeftMerge127759
def owner : Owner := ⟨.program ⟨257⟩, ⟨19112⟩⟩
def mergeEvent : Nat := 127759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19685⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127754RawTerms
def group : MergeGroup := .relation 127756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127756) (rhsResult := 127754)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127755 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩]⟩) (none) 127754) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19685⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], [⟨.program ⟨257⟩, ⟨19685⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127759

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
