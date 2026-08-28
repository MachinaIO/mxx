import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge3803
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 9)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3803

namespace LeftMerge3804
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3804
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 11)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3804

namespace LeftMerge3805
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3805
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 12)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3805

namespace LeftMerge3806
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 13)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3806

namespace LeftMerge3807
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3807
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 15)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29380⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3807

namespace LeftMerge3808
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3808
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 16)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26700⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3808

namespace LeftMerge3809
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3809
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 18)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67008⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3809

namespace LeftMerge3810
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 0)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨63199⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3810

namespace LeftMerge3811
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 1)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60219⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3811

namespace LeftMerge3812
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 2)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57239⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3812

namespace LeftMerge3813
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 3)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54259⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3813

namespace LeftMerge3814
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 4)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3814

namespace LeftMerge3815
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3815
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 6)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3815

namespace LeftMerge3816
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3816
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 10)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3816

namespace LeftMerge3817
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3817
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 14)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3817

namespace LeftMerge3818
def owner : Owner := ⟨.program ⟨257⟩, ⟨67590⟩⟩
def mergeEvent : Nat := 3818
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3796RawTerms
def rightRaw : List Term := Proof.Events012.exact3073RawTerms
def group : MergeGroup := .operator 3796 3073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3796) (leftOrdinal := 17)
    (rightResult := 3073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6733⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6733⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3818

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
