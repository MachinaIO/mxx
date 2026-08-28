import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge7545
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 12)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7545

namespace LeftMerge7546
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 13)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7546

namespace LeftMerge7547
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 15)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7547

namespace LeftMerge7548
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7548
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 16)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26583⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7548

namespace LeftMerge7549
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7549
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 18)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66378⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7549

namespace LeftMerge7550
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7550
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 0)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7550

namespace LeftMerge7551
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 1)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨60048⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7551

namespace LeftMerge7552
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 2)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨57068⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7552

namespace LeftMerge7553
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 3)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7553

namespace LeftMerge7554
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 4)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨51108⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7554

namespace LeftMerge7555
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7555
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 6)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32044⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7555

namespace LeftMerge7556
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7556
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 10)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22024⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7556

namespace LeftMerge7557
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7557
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 14)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18804⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7557

namespace LeftMerge7558
def owner : Owner := ⟨.program ⟨257⟩, ⟨67403⟩⟩
def mergeEvent : Nat := 7558
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events029.exact7536RawTerms
def rightRaw : List Term := Proof.Events026.exact6813RawTerms
def group : MergeGroup := .operator 7536 6813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7536) (leftOrdinal := 17)
    (rightResult := 6813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6771⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6771⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15982⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7558

namespace LeftMerge8063
def owner : Owner := ⟨.program ⟨257⟩, ⟨67539⟩⟩
def mergeEvent : Nat := 8063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events031.exact8059RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 8059 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8059) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67538⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8063

namespace LeftMerge8071
def owner : Owner := ⟨.program ⟨257⟩, ⟨48412⟩⟩
def mergeEvent : Nat := 8071
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events031.exact8067RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 8067 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8067) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48411⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8071

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
