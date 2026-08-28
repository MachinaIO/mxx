import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge175855
def owner : Owner := ⟨.program ⟨257⟩, ⟨70482⟩⟩
def mergeEvent : Nat := 175855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩] } }
def leftRaw : List Term := Proof.Events686.exact175848RawTerms
def rightRaw : List Term := Proof.Events061.exact15702RawTerms
def group : MergeGroup := .operator 175848 15702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175848) (leftOrdinal := 1)
    (rightResult := 15702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7173⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175855

namespace LeftMerge175857
def owner : Owner := ⟨.program ⟨257⟩, ⟨70482⟩⟩
def mergeEvent : Nat := 175857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15695RawTerms
def group : MergeGroup := .relation 175856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175856) (rhsResult := 15695)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7173⟩⟩) ⟨7052⟩ 15695) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6870⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175857

namespace LeftMerge175871
def owner : Owner := ⟨.program ⟨257⟩, ⟨64991⟩⟩
def mergeEvent : Nat := 175871
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168269RawTerms
def rightRaw : List Term := Proof.Events686.exact175865RawTerms
def group : MergeGroup := .operator 168269 175865
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168269) (leftOrdinal := 0)
    (rightResult := 175865) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175871

namespace LeftMerge175872
def owner : Owner := ⟨.program ⟨257⟩, ⟨64991⟩⟩
def mergeEvent : Nat := 175872
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩] } }
def leftRaw : List Term := Proof.Events657.exact168269RawTerms
def rightRaw : List Term := Proof.Events686.exact175865RawTerms
def group : MergeGroup := .operator 168269 175865
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 168269) (leftOrdinal := 1)
    (rightResult := 175865) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175872

namespace LeftMerge175874
def owner : Owner := ⟨.program ⟨257⟩, ⟨64991⟩⟩
def mergeEvent : Nat := 175874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64116⟩⟩] } }
def rhsRaw : List Term := Proof.Events686.exact175862RawTerms
def group : MergeGroup := .relation 175873
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175873) (rhsResult := 175862)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64989⟩⟩) ⟨64116⟩ 175862) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64116⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨64116⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175874

namespace LeftMerge175888
def owner : Owner := ⟨.program ⟨257⟩, ⟨63755⟩⟩
def mergeEvent : Nat := 175888
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events687.exact175882RawTerms
def group : MergeGroup := .operator 163745 175882
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 175882) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63752⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175888

namespace LeftMerge176009
def owner : Owner := ⟨.program ⟨257⟩, ⟨64304⟩⟩
def mergeEvent : Nat := 176009
def frameStart : Nat := 175943
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events687.exact176005RawTerms
def rightRaw : List Term := Proof.Events687.exact176003RawTerms
def group : MergeGroup := .operator 176005 176003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176005) (leftOrdinal := 0)
    (rightResult := 176003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176009

namespace LeftMerge176021
def owner : Owner := ⟨.program ⟨257⟩, ⟨64990⟩⟩
def mergeEvent : Nat := 176021
def frameStart : Nat := 175943
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩] } }
def leftRaw : List Term := Proof.Events687.exact176017RawTerms
def rightRaw : List Term := Proof.Events687.exact175994RawTerms
def group : MergeGroup := .operator 176017 175994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176017) (leftOrdinal := 0)
    (rightResult := 175994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64989⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176021

namespace LeftMerge176022
def owner : Owner := ⟨.program ⟨257⟩, ⟨64990⟩⟩
def mergeEvent : Nat := 176022
def frameStart : Nat := 175943
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩] } }
def leftRaw : List Term := Proof.Events687.exact176017RawTerms
def rightRaw : List Term := Proof.Events687.exact175994RawTerms
def group : MergeGroup := .operator 176017 175994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176017) (leftOrdinal := 1)
    (rightResult := 175994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176022

namespace LeftMerge176024
def owner : Owner := ⟨.program ⟨257⟩, ⟨64990⟩⟩
def mergeEvent : Nat := 176024
def frameStart : Nat := 175943
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64116⟩⟩] } }
def rhsRaw : List Term := Proof.Events687.exact175991RawTerms
def group : MergeGroup := .relation 176023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176023) (rhsResult := 175991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64989⟩⟩) ⟨64116⟩ 175991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64116⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨64116⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176024

namespace LeftMerge176032
def owner : Owner := ⟨.program ⟨257⟩, ⟨63164⟩⟩
def mergeEvent : Nat := 176032
def frameStart : Nat := 175943
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events687.exact176005RawTerms
def rightRaw : List Term := Proof.Events687.exact176028RawTerms
def group : MergeGroup := .operator 176005 176028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176005) (leftOrdinal := 0)
    (rightResult := 176028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63161⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176032

namespace LeftMerge176049
def owner : Owner := ⟨.program ⟨257⟩, ⟨63755⟩⟩
def mergeEvent : Nat := 176049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩] } }
def rhsRaw : List Term := Proof.Events687.exact176046RawTerms
def group : MergeGroup := .relation 176048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176048) (rhsResult := 176046)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 176047 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩]⟩) (none) 176046) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176049

namespace LeftMerge176050
def owner : Owner := ⟨.program ⟨257⟩, ⟨63755⟩⟩
def mergeEvent : Nat := 176050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩] } }
def rhsRaw : List Term := Proof.Events687.exact176046RawTerms
def group : MergeGroup := .relation 176048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176048) (rhsResult := 176046)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 176047 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩]⟩) (none) 176046) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176050

namespace LeftMerge176051
def owner : Owner := ⟨.program ⟨257⟩, ⟨63755⟩⟩
def mergeEvent : Nat := 176051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64116⟩⟩] } }
def rhsRaw : List Term := Proof.Events687.exact176046RawTerms
def group : MergeGroup := .relation 176048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176048) (rhsResult := 176046)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 176047 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩]⟩) (none) 176046) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64116⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨62840⟩⟩], [⟨.program ⟨257⟩, ⟨64116⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176051

namespace LeftMerge176052
def owner : Owner := ⟨.program ⟨257⟩, ⟨63755⟩⟩
def mergeEvent : Nat := 176052
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events687.exact176046RawTerms
def group : MergeGroup := .relation 176048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 176048) (rhsResult := 176046)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 176047 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63752⟩⟩]⟩) (none) 176046) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63161⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨63161⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge176052

namespace LeftMerge176057
def owner : Owner := ⟨.program ⟨257⟩, ⟨64992⟩⟩
def mergeEvent : Nat := 176057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩] } }
def leftRaw : List Term := Proof.Events687.exact176053RawTerms
def rightRaw : List Term := Proof.Events687.exact175875RawTerms
def group : MergeGroup := .operator 176053 175875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 176053) (leftOrdinal := 0)
    (rightResult := 175875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64989⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge176057

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
