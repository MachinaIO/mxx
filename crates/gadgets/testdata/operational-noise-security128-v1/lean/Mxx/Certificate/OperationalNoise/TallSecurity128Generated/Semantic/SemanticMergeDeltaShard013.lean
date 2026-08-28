import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge4451
def owner : Owner := ⟨.program ⟨257⟩, ⟨22177⟩⟩
def mergeEvent : Nat := 4451
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4447RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 4447 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4447) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22176⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4451

namespace LeftMerge4459
def owner : Owner := ⟨.program ⟨257⟩, ⟨18957⟩⟩
def mergeEvent : Nat := 4459
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4455RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 4455 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4455) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18956⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4459

namespace LeftMerge4467
def owner : Owner := ⟨.program ⟨257⟩, ⟨16111⟩⟩
def mergeEvent : Nat := 4467
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4463RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 4463 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4463) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16110⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4467

namespace LeftMerge4548
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4548
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 5)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge4548

namespace LeftMerge4549
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4549
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 7)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4549

namespace LeftMerge4550
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4550
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 8)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4550

namespace LeftMerge4551
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 9)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4551

namespace LeftMerge4552
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 11)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4552

namespace LeftMerge4553
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 12)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4553

namespace LeftMerge4554
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 13)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4554

namespace LeftMerge4555
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4555
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 15)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4555

namespace LeftMerge4556
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4556
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 16)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4556

namespace LeftMerge4557
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4557
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 18)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66938⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4557

namespace LeftMerge4558
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4558
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 0)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨63180⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4558

namespace LeftMerge4559
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4559
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 1)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨60200⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4559

namespace LeftMerge4560
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def mergeEvent : Nat := 4560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events017.exact4544RawTerms
def rightRaw : List Term := Proof.Events014.exact3821RawTerms
def group : MergeGroup := .operator 4544 3821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4544) (leftOrdinal := 2)
    (rightResult := 3821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6755⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6755⟩⟩, ⟨.program ⟨257⟩, ⟨57220⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge4560

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
