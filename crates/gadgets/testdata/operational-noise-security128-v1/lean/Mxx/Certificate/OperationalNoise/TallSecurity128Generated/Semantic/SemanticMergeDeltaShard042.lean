import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge13529
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 12)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37532⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13529

namespace LeftMerge13530
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13530
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 13)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34852⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13530

namespace LeftMerge13531
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13531
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 15)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29195⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13531

namespace LeftMerge13532
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13532
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 16)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26515⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13532

namespace LeftMerge13533
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 18)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66006⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13533

namespace LeftMerge13534
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 0)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨62928⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13534

namespace LeftMerge13535
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 1)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨59948⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13535

namespace LeftMerge13536
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 2)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨56968⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13536

namespace LeftMerge13537
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 3)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨53988⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13537

namespace LeftMerge13538
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 4)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨51008⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13538

namespace LeftMerge13539
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 6)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨31944⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13539

namespace LeftMerge13540
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 10)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨21924⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13540

namespace LeftMerge13541
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 14)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18704⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13541

namespace LeftMerge13542
def owner : Owner := ⟨.program ⟨257⟩, ⟨67304⟩⟩
def mergeEvent : Nat := 13542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events052.exact13520RawTerms
def rightRaw : List Term := Proof.Events049.exact12797RawTerms
def group : MergeGroup := .operator 13520 12797
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13520) (leftOrdinal := 17)
    (rightResult := 12797) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6826⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6826⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15898⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13542

namespace LeftMerge14041
def owner : Owner := ⟨.program ⟨257⟩, ⟨67342⟩⟩
def mergeEvent : Nat := 14041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events054.exact14037RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 14037 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14037) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67341⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67341⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14041

namespace LeftMerge14049
def owner : Owner := ⟨.program ⟨257⟩, ⟨48282⟩⟩
def mergeEvent : Nat := 14049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events054.exact14045RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 14045 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14045) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48281⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48281⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14049

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
