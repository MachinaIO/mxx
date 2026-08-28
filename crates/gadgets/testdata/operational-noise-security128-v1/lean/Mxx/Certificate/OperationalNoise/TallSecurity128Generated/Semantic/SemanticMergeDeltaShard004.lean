import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge1561
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 12)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1561

namespace LeftMerge1562
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35076⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 13)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35076⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35076⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1562

namespace LeftMerge1563
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29419⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 15)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29419⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29419⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1563

namespace LeftMerge1564
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26739⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 16)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26739⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26739⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1564

namespace LeftMerge1565
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67218⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 18)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67218⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67218⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1565

namespace LeftMerge1566
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1566
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨63256⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 0)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63256⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨63256⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1566

namespace LeftMerge1567
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1567
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨60276⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 1)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60276⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨60276⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1567

namespace LeftMerge1568
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1568
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨57296⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 2)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57296⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨57296⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1568

namespace LeftMerge1569
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1569
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 3)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54316⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1569

namespace LeftMerge1570
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1570
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 4)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51336⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1570

namespace LeftMerge1571
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32272⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 6)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32272⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32272⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1571

namespace LeftMerge1572
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1572
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22252⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 10)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22252⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22252⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1572

namespace LeftMerge1573
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1573
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 14)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨19032⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1573

namespace LeftMerge1574
def owner : Owner := ⟨.program ⟨257⟩, ⟨67651⟩⟩
def mergeEvent : Nat := 1574
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16174⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events006.exact1552RawTerms
def rightRaw : List Term := Proof.Events003.exact829RawTerms
def group : MergeGroup := .operator 1552 829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1552) (leftOrdinal := 17)
    (rightResult := 829) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16174⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6746⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6746⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16174⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge1574

namespace LeftMerge2079
def owner : Owner := ⟨.program ⟨257⟩, ⟨67627⟩⟩
def mergeEvent : Nat := 2079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2075RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 2075 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2075) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67626⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67626⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2079

namespace LeftMerge2087
def owner : Owner := ⟨.program ⟨257⟩, ⟨48464⟩⟩
def mergeEvent : Nat := 2087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events008.exact2083RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 2083 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2083) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48463⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48463⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge2087

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
