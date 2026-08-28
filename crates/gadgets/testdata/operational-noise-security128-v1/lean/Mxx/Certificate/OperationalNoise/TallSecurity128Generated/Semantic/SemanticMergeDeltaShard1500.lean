import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge243776
def owner : Owner := ⟨.program ⟨257⟩, ⟨31814⟩⟩
def mergeEvent : Nat := 243776
def frameStart : Nat := 243673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243729RawTerms
def rightRaw : List Term := Proof.Events952.exact243772RawTerms
def group : MergeGroup := .operator 243729 243772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243729) (leftOrdinal := 0)
    (rightResult := 243772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243776

namespace LeftMerge243793
def owner : Owner := ⟨.program ⟨257⟩, ⟨32372⟩⟩
def mergeEvent : Nat := 243793
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events952.exact243790RawTerms
def group : MergeGroup := .relation 243792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243792) (rhsResult := 243790)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩]⟩) (none) 243790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243793

namespace LeftMerge243794
def owner : Owner := ⟨.program ⟨257⟩, ⟨32372⟩⟩
def mergeEvent : Nat := 243794
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩] } }
def rhsRaw : List Term := Proof.Events952.exact243790RawTerms
def group : MergeGroup := .relation 243792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243792) (rhsResult := 243790)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩]⟩) (none) 243790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243794

namespace LeftMerge243795
def owner : Owner := ⟨.program ⟨257⟩, ⟨32372⟩⟩
def mergeEvent : Nat := 243795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32937⟩⟩] } }
def rhsRaw : List Term := Proof.Events952.exact243790RawTerms
def group : MergeGroup := .relation 243792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243792) (rhsResult := 243790)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩]⟩) (none) 243790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32937⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨32937⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243795

namespace LeftMerge243796
def owner : Owner := ⟨.program ⟨257⟩, ⟨32372⟩⟩
def mergeEvent : Nat := 243796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events952.exact243790RawTerms
def group : MergeGroup := .relation 243792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243792) (rhsResult := 243790)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 243791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32369⟩⟩]⟩) (none) 243790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243796

namespace LeftMerge243801
def owner : Owner := ⟨.program ⟨257⟩, ⟨33439⟩⟩
def mergeEvent : Nat := 243801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32937⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243797RawTerms
def rightRaw : List Term := Proof.Events951.exact243611RawTerms
def group : MergeGroup := .operator 243797 243611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243797) (leftOrdinal := 2)
    (rightResult := 243611) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32937⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32937⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24266⟩⟩, ⟨.program ⟨257⟩, ⟨31431⟩⟩], [⟨.program ⟨257⟩, ⟨32937⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243801

namespace LeftMerge243802
def owner : Owner := ⟨.program ⟨257⟩, ⟨33439⟩⟩
def mergeEvent : Nat := 243802
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243797RawTerms
def rightRaw : List Term := Proof.Events951.exact243611RawTerms
def group : MergeGroup := .operator 243797 243611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243797) (leftOrdinal := 1)
    (rightResult := 243611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243802

namespace LeftMerge243810
def owner : Owner := ⟨.program ⟨257⟩, ⟨33832⟩⟩
def mergeEvent : Nat := 243810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243804RawTerms
def rightRaw : List Term := Proof.Events951.exact243527RawTerms
def group : MergeGroup := .operator 243804 243527
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243804) (leftOrdinal := 0)
    (rightResult := 243527) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243810

namespace LeftMerge243811
def owner : Owner := ⟨.program ⟨257⟩, ⟨33832⟩⟩
def mergeEvent : Nat := 243811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243804RawTerms
def rightRaw : List Term := Proof.Events951.exact243527RawTerms
def group : MergeGroup := .operator 243804 243527
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243804) (leftOrdinal := 1)
    (rightResult := 243527) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243811

namespace LeftMerge243813
def owner : Owner := ⟨.program ⟨257⟩, ⟨33832⟩⟩
def mergeEvent : Nat := 243813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33083⟩⟩] } }
def rhsRaw : List Term := Proof.Events951.exact243524RawTerms
def group : MergeGroup := .relation 243812
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243812) (rhsResult := 243524)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33830⟩⟩) ⟨33083⟩ 243524) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33083⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨31812⟩⟩], [⟨.program ⟨257⟩, ⟨33083⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243813

namespace LeftMerge243827
def owner : Owner := ⟨.program ⟨257⟩, ⟨32659⟩⟩
def mergeEvent : Nat := 243827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32656⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events952.exact243821RawTerms
def group : MergeGroup := .operator 236870 243821
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 243821) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32656⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32656⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243827

namespace LeftMerge243948
def owner : Owner := ⟨.program ⟨257⟩, ⟨33300⟩⟩
def mergeEvent : Nat := 243948
def frameStart : Nat := 243882
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243944RawTerms
def rightRaw : List Term := Proof.Events952.exact243942RawTerms
def group : MergeGroup := .operator 243944 243942
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243944) (leftOrdinal := 0)
    (rightResult := 243942) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243948

namespace LeftMerge243960
def owner : Owner := ⟨.program ⟨257⟩, ⟨33831⟩⟩
def mergeEvent : Nat := 243960
def frameStart : Nat := 243882
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243956RawTerms
def rightRaw : List Term := Proof.Events952.exact243933RawTerms
def group : MergeGroup := .operator 243956 243933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243956) (leftOrdinal := 0)
    (rightResult := 243933) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33830⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243960

namespace LeftMerge243961
def owner : Owner := ⟨.program ⟨257⟩, ⟨33831⟩⟩
def mergeEvent : Nat := 243961
def frameStart : Nat := 243882
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243956RawTerms
def rightRaw : List Term := Proof.Events952.exact243933RawTerms
def group : MergeGroup := .operator 243956 243933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243956) (leftOrdinal := 1)
    (rightResult := 243933) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33830⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243961

namespace LeftMerge243963
def owner : Owner := ⟨.program ⟨257⟩, ⟨33831⟩⟩
def mergeEvent : Nat := 243963
def frameStart : Nat := 243882
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33083⟩⟩] } }
def rhsRaw : List Term := Proof.Events952.exact243930RawTerms
def group : MergeGroup := .relation 243962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 243962) (rhsResult := 243930)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33830⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33830⟩⟩) ⟨33083⟩ 243930) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33083⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31812⟩⟩], [⟨.program ⟨257⟩, ⟨33083⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge243963

namespace LeftMerge243971
def owner : Owner := ⟨.program ⟨257⟩, ⟨32070⟩⟩
def mergeEvent : Nat := 243971
def frameStart : Nat := 243882
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events952.exact243944RawTerms
def rightRaw : List Term := Proof.Events952.exact243967RawTerms
def group : MergeGroup := .operator 243944 243967
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243944) (leftOrdinal := 0)
    (rightResult := 243967) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32068⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge243971

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
