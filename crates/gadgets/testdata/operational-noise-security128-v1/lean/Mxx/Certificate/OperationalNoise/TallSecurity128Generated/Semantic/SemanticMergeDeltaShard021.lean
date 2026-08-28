import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge6803
def owner : Owner := ⟨.program ⟨257⟩, ⟨67326⟩⟩
def mergeEvent : Nat := 6803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events026.exact6788RawTerms
def rightRaw : List Term := Proof.Events023.exact6065RawTerms
def group : MergeGroup := .operator 6788 6065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6788) (leftOrdinal := 1)
    (rightResult := 6065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨59972⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6803

namespace LeftMerge6804
def owner : Owner := ⟨.program ⟨257⟩, ⟨67326⟩⟩
def mergeEvent : Nat := 6804
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events026.exact6788RawTerms
def rightRaw : List Term := Proof.Events023.exact6065RawTerms
def group : MergeGroup := .operator 6788 6065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6788) (leftOrdinal := 2)
    (rightResult := 6065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨56992⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6804

namespace LeftMerge6805
def owner : Owner := ⟨.program ⟨257⟩, ⟨67326⟩⟩
def mergeEvent : Nat := 6805
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events026.exact6788RawTerms
def rightRaw : List Term := Proof.Events023.exact6065RawTerms
def group : MergeGroup := .operator 6788 6065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6788) (leftOrdinal := 3)
    (rightResult := 6065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54012⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6805

namespace LeftMerge6806
def owner : Owner := ⟨.program ⟨257⟩, ⟨67326⟩⟩
def mergeEvent : Nat := 6806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events026.exact6788RawTerms
def rightRaw : List Term := Proof.Events023.exact6065RawTerms
def group : MergeGroup := .operator 6788 6065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6788) (leftOrdinal := 4)
    (rightResult := 6065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51032⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6806

namespace LeftMerge6807
def owner : Owner := ⟨.program ⟨257⟩, ⟨67326⟩⟩
def mergeEvent : Nat := 6807
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events026.exact6788RawTerms
def rightRaw : List Term := Proof.Events023.exact6065RawTerms
def group : MergeGroup := .operator 6788 6065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6788) (leftOrdinal := 6)
    (rightResult := 6065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6807

namespace LeftMerge6808
def owner : Owner := ⟨.program ⟨257⟩, ⟨67326⟩⟩
def mergeEvent : Nat := 6808
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events026.exact6788RawTerms
def rightRaw : List Term := Proof.Events023.exact6065RawTerms
def group : MergeGroup := .operator 6788 6065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6788) (leftOrdinal := 10)
    (rightResult := 6065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6808

namespace LeftMerge6809
def owner : Owner := ⟨.program ⟨257⟩, ⟨67326⟩⟩
def mergeEvent : Nat := 6809
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events026.exact6788RawTerms
def rightRaw : List Term := Proof.Events023.exact6065RawTerms
def group : MergeGroup := .operator 6788 6065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6788) (leftOrdinal := 14)
    (rightResult := 6065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6809

namespace LeftMerge6810
def owner : Owner := ⟨.program ⟨257⟩, ⟨67326⟩⟩
def mergeEvent : Nat := 6810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events026.exact6788RawTerms
def rightRaw : List Term := Proof.Events023.exact6065RawTerms
def group : MergeGroup := .operator 6788 6065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6788) (leftOrdinal := 17)
    (rightResult := 6065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6810

namespace LeftMerge7315
def owner : Owner := ⟨.program ⟨257⟩, ⟨67400⟩⟩
def mergeEvent : Nat := 7315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events028.exact7311RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 7311 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7311) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7315

namespace LeftMerge7323
def owner : Owner := ⟨.program ⟨257⟩, ⟨48321⟩⟩
def mergeEvent : Nat := 7323
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events028.exact7319RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 7319 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7319) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48320⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7323

namespace LeftMerge7331
def owner : Owner := ⟨.program ⟨257⟩, ⟨45641⟩⟩
def mergeEvent : Nat := 7331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events028.exact7327RawTerms
def rightRaw : List Term := Proof.Events002.exact553RawTerms
def group : MergeGroup := .operator 7327 553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7327) (leftOrdinal := 0)
    (rightResult := 553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45640⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7331

namespace LeftMerge7339
def owner : Owner := ⟨.program ⟨257⟩, ⟨42964⟩⟩
def mergeEvent : Nat := 7339
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events028.exact7335RawTerms
def rightRaw : List Term := Proof.Events002.exact563RawTerms
def group : MergeGroup := .operator 7335 563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7335) (leftOrdinal := 0)
    (rightResult := 563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42963⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7339

namespace LeftMerge7347
def owner : Owner := ⟨.program ⟨257⟩, ⟨40284⟩⟩
def mergeEvent : Nat := 7347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events028.exact7343RawTerms
def rightRaw : List Term := Proof.Events002.exact573RawTerms
def group : MergeGroup := .operator 7343 573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7343) (leftOrdinal := 0)
    (rightResult := 573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40283⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7347

namespace LeftMerge7355
def owner : Owner := ⟨.program ⟨257⟩, ⟨37601⟩⟩
def mergeEvent : Nat := 7355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events028.exact7351RawTerms
def rightRaw : List Term := Proof.Events002.exact583RawTerms
def group : MergeGroup := .operator 7351 583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7351) (leftOrdinal := 0)
    (rightResult := 583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7355

namespace LeftMerge7363
def owner : Owner := ⟨.program ⟨257⟩, ⟨34921⟩⟩
def mergeEvent : Nat := 7363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events028.exact7359RawTerms
def rightRaw : List Term := Proof.Events002.exact593RawTerms
def group : MergeGroup := .operator 7359 593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7359) (leftOrdinal := 0)
    (rightResult := 593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34920⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨34920⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7363

namespace LeftMerge7371
def owner : Owner := ⟨.program ⟨257⟩, ⟨29264⟩⟩
def mergeEvent : Nat := 7371
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events028.exact7367RawTerms
def rightRaw : List Term := Proof.Events002.exact603RawTerms
def group : MergeGroup := .operator 7367 603
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7367) (leftOrdinal := 0)
    (rightResult := 603) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29263⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29263⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7371

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
