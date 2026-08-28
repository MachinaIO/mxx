import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge12663
def owner : Owner := ⟨.program ⟨257⟩, ⟨51071⟩⟩
def mergeEvent : Nat := 12663
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12659RawTerms
def rightRaw : List Term := Proof.Events002.exact673RawTerms
def group : MergeGroup := .operator 12659 673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12659) (leftOrdinal := 0)
    (rightResult := 673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51070⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12663

namespace LeftMerge12671
def owner : Owner := ⟨.program ⟨257⟩, ⟨32007⟩⟩
def mergeEvent : Nat := 12671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12667RawTerms
def rightRaw : List Term := Proof.Events002.exact683RawTerms
def group : MergeGroup := .operator 12667 683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12667) (leftOrdinal := 0)
    (rightResult := 683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32006⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12671

namespace LeftMerge12679
def owner : Owner := ⟨.program ⟨257⟩, ⟨21987⟩⟩
def mergeEvent : Nat := 12679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12675RawTerms
def rightRaw : List Term := Proof.Events002.exact693RawTerms
def group : MergeGroup := .operator 12675 693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12675) (leftOrdinal := 0)
    (rightResult := 693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21986⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21986⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12679

namespace LeftMerge12687
def owner : Owner := ⟨.program ⟨257⟩, ⟨18767⟩⟩
def mergeEvent : Nat := 12687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12683RawTerms
def rightRaw : List Term := Proof.Events002.exact703RawTerms
def group : MergeGroup := .operator 12683 703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12683) (leftOrdinal := 0)
    (rightResult := 703) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18766⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18766⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12687

namespace LeftMerge12695
def owner : Owner := ⟨.program ⟨257⟩, ⟨15951⟩⟩
def mergeEvent : Nat := 12695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12691RawTerms
def rightRaw : List Term := Proof.Events002.exact713RawTerms
def group : MergeGroup := .operator 12691 713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12691) (leftOrdinal := 0)
    (rightResult := 713) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15950⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15950⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12695

namespace LeftMerge12776
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 5)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67363⟩⟩], []⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge12776

namespace LeftMerge12777
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 7)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48294⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12777

namespace LeftMerge12778
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 8)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45614⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12778

namespace LeftMerge12779
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 9)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42937⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12779

namespace LeftMerge12780
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12780
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 11)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40257⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12780

namespace LeftMerge12781
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12781
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 12)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37574⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12781

namespace LeftMerge12782
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12782
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 13)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34894⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12782

namespace LeftMerge12783
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12783
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 15)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29237⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12783

namespace LeftMerge12784
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12784
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 16)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12784

namespace LeftMerge12785
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12785
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 18)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12785

namespace LeftMerge12786
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def mergeEvent : Nat := 12786
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events049.exact12772RawTerms
def rightRaw : List Term := Proof.Events047.exact12049RawTerms
def group : MergeGroup := .operator 12772 12049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12772) (leftOrdinal := 0)
    (rightResult := 12049) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6739⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6739⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge12786

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
