import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge3061
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 18)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3061

namespace LeftMerge3062
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 0)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3062

namespace LeftMerge3063
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 1)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3063

namespace LeftMerge3064
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3064
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 2)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3064

namespace LeftMerge3065
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3065
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 3)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3065

namespace LeftMerge3066
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3066
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 4)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3066

namespace LeftMerge3067
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3067
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 6)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32234⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3067

namespace LeftMerge3068
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3068
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 10)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22214⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3068

namespace LeftMerge3069
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3069
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 14)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18994⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3069

namespace LeftMerge3070
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def mergeEvent : Nat := 3070
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events011.exact3048RawTerms
def rightRaw : List Term := Proof.Events009.exact2325RawTerms
def group : MergeGroup := .operator 3048 2325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3048) (leftOrdinal := 17)
    (rightResult := 2325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6779⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6779⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16142⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3070

namespace LeftMerge3575
def owner : Owner := ⟨.program ⟨257⟩, ⟨67587⟩⟩
def mergeEvent : Nat := 3575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events013.exact3571RawTerms
def rightRaw : List Term := Proof.Events000.exact36RawTerms
def group : MergeGroup := .operator 3571 36
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3571) (leftOrdinal := 0)
    (rightResult := 36) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67586⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3575

namespace LeftMerge3583
def owner : Owner := ⟨.program ⟨257⟩, ⟨48438⟩⟩
def mergeEvent : Nat := 3583
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events013.exact3579RawTerms
def rightRaw : List Term := Proof.Events002.exact543RawTerms
def group : MergeGroup := .operator 3579 543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3579) (leftOrdinal := 0)
    (rightResult := 543) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48437⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3583

namespace LeftMerge3591
def owner : Owner := ⟨.program ⟨257⟩, ⟨45758⟩⟩
def mergeEvent : Nat := 3591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3587RawTerms
def rightRaw : List Term := Proof.Events002.exact553RawTerms
def group : MergeGroup := .operator 3587 553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3587) (leftOrdinal := 0)
    (rightResult := 553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45757⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3591

namespace LeftMerge3599
def owner : Owner := ⟨.program ⟨257⟩, ⟨43081⟩⟩
def mergeEvent : Nat := 3599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3595RawTerms
def rightRaw : List Term := Proof.Events002.exact563RawTerms
def group : MergeGroup := .operator 3595 563
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3595) (leftOrdinal := 0)
    (rightResult := 563) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3599

namespace LeftMerge3607
def owner : Owner := ⟨.program ⟨257⟩, ⟨40401⟩⟩
def mergeEvent : Nat := 3607
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3603RawTerms
def rightRaw : List Term := Proof.Events002.exact573RawTerms
def group : MergeGroup := .operator 3603 573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3603) (leftOrdinal := 0)
    (rightResult := 573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40400⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3607

namespace LeftMerge3615
def owner : Owner := ⟨.program ⟨257⟩, ⟨37718⟩⟩
def mergeEvent : Nat := 3615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events014.exact3611RawTerms
def rightRaw : List Term := Proof.Events002.exact583RawTerms
def group : MergeGroup := .operator 3611 583
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3611) (leftOrdinal := 0)
    (rightResult := 583) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge3615

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
