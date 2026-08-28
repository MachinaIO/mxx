import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge127045
def owner : Owner := ⟨.program ⟨257⟩, ⟨21404⟩⟩
def mergeEvent : Nat := 127045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events496.exact127039RawTerms
def rightRaw : List Term := Proof.Events022.exact5678RawTerms
def group : MergeGroup := .operator 127039 5678
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127039) (leftOrdinal := 1)
    (rightResult := 5678) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127045

namespace LeftMerge127046
def owner : Owner := ⟨.program ⟨257⟩, ⟨21404⟩⟩
def mergeEvent : Nat := 127046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events496.exact127039RawTerms
def rightRaw : List Term := Proof.Events022.exact5678RawTerms
def group : MergeGroup := .operator 127039 5678
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127039) (leftOrdinal := 0)
    (rightResult := 5678) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127046

namespace LeftMerge127051
def owner : Owner := ⟨.program ⟨257⟩, ⟨21042⟩⟩
def mergeEvent : Nat := 127051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events022.exact5678RawTerms
def rightRaw : List Term := Proof.Events467.exact119778RawTerms
def group : MergeGroup := .operator 5678 119778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5678) (leftOrdinal := 0)
    (rightResult := 119778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127051

namespace LeftMerge127056
def owner : Owner := ⟨.program ⟨257⟩, ⟨8136⟩⟩
def mergeEvent : Nat := 127056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }
def leftRaw : List Term := Proof.Events467.exact119648RawTerms
def rightRaw : List Term := Proof.Events096.exact24636RawTerms
def group : MergeGroup := .operator 119648 24636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119648) (leftOrdinal := 0)
    (rightResult := 24636) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127056

namespace LeftMerge127073
def owner : Owner := ⟨.program ⟨257⟩, ⟨21045⟩⟩
def mergeEvent : Nat := 127073
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events496.exact127067RawTerms
def rightRaw : List Term := Proof.Events096.exact24625RawTerms
def group : MergeGroup := .operator 127067 24625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127067) (leftOrdinal := 1)
    (rightResult := 24625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127073

namespace LeftMerge127075
def owner : Owner := ⟨.program ⟨257⟩, ⟨21045⟩⟩
def mergeEvent : Nat := 127075
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24595RawTerms
def group : MergeGroup := .relation 127074
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127074) (rhsResult := 24595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9574⟩⟩) ⟨7306⟩ 24595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127075

namespace LeftMerge127076
def owner : Owner := ⟨.program ⟨257⟩, ⟨21045⟩⟩
def mergeEvent : Nat := 127076
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events496.exact127067RawTerms
def rightRaw : List Term := Proof.Events096.exact24625RawTerms
def group : MergeGroup := .operator 127067 24625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127067) (leftOrdinal := 0)
    (rightResult := 24625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127076

namespace LeftMerge127081
def owner : Owner := ⟨.program ⟨257⟩, ⟨21405⟩⟩
def mergeEvent : Nat := 127081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events496.exact127077RawTerms
def rightRaw : List Term := Proof.Events496.exact127047RawTerms
def group : MergeGroup := .operator 127077 127047
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127077) (leftOrdinal := 1)
    (rightResult := 127047) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127081

namespace LeftMerge127089
def owner : Owner := ⟨.program ⟨257⟩, ⟨23396⟩⟩
def mergeEvent : Nat := 127089
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩] } }
def leftRaw : List Term := Proof.Events496.exact127083RawTerms
def rightRaw : List Term := Proof.Events496.exact127019RawTerms
def group : MergeGroup := .operator 127083 127019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127083) (leftOrdinal := 1)
    (rightResult := 127019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127089

namespace LeftMerge127091
def owner : Owner := ⟨.program ⟨257⟩, ⟨23396⟩⟩
def mergeEvent : Nat := 127091
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22905⟩⟩] } }
def rhsRaw : List Term := Proof.Events496.exact127016RawTerms
def group : MergeGroup := .relation 127090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127090) (rhsResult := 127016)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23395⟩⟩) ⟨22905⟩ 127016) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22905⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨22905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127091

namespace LeftMerge127092
def owner : Owner := ⟨.program ⟨257⟩, ⟨23396⟩⟩
def mergeEvent : Nat := 127092
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩] } }
def leftRaw : List Term := Proof.Events496.exact127083RawTerms
def rightRaw : List Term := Proof.Events496.exact127019RawTerms
def group : MergeGroup := .operator 127083 127019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127083) (leftOrdinal := 0)
    (rightResult := 127019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127092

namespace LeftMerge127106
def owner : Owner := ⟨.program ⟨257⟩, ⟨22332⟩⟩
def mergeEvent : Nat := 127106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events496.exact127100RawTerms
def group : MergeGroup := .operator 119870 127100
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 127100) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127106

namespace LeftMerge127185
def owner : Owner := ⟨.program ⟨257⟩, ⟨21399⟩⟩
def mergeEvent : Nat := 127185
def frameStart : Nat := 127155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events496.exact127181RawTerms
def rightRaw : List Term := Proof.Events496.exact127178RawTerms
def group : MergeGroup := .operator 127181 127178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127181) (leftOrdinal := 0)
    (rightResult := 127178) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127185

namespace LeftMerge127215
def owner : Owner := ⟨.program ⟨257⟩, ⟨23192⟩⟩
def mergeEvent : Nat := 127215
def frameStart : Nat := 127155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events496.exact127211RawTerms
def rightRaw : List Term := Proof.Events496.exact127209RawTerms
def group : MergeGroup := .operator 127211 127209
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127211) (leftOrdinal := 0)
    (rightResult := 127209) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21041⟩⟩, ⟨.program ⟨257⟩, ⟨21398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127215

namespace LeftMerge127238
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 127238
def frameStart : Nat := 127155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127234RawTerms
def rightRaw : List Term := Proof.Events496.exact127231RawTerms
def group : MergeGroup := .operator 127234 127231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127234) (leftOrdinal := 0)
    (rightResult := 127231) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127238

namespace LeftMerge127247
def owner : Owner := ⟨.program ⟨257⟩, ⟨23398⟩⟩
def mergeEvent : Nat := 127247
def frameStart : Nat := 127155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩] } }
def leftRaw : List Term := Proof.Events497.exact127243RawTerms
def rightRaw : List Term := Proof.Events496.exact127200RawTerms
def group : MergeGroup := .operator 127243 127200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127243) (leftOrdinal := 0)
    (rightResult := 127200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23395⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23395⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127247

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
