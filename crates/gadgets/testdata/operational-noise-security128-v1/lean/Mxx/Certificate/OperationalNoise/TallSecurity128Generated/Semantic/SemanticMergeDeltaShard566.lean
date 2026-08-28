import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge94624
def owner : Owner := ⟨.program ⟨257⟩, ⟨69298⟩⟩
def mergeEvent : Nat := 94624
def frameStart : Nat := 94531
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94619RawTerms
def rightRaw : List Term := Proof.Events369.exact94576RawTerms
def group : MergeGroup := .operator 94619 94576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94619) (leftOrdinal := 1)
    (rightResult := 94576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69295⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94624

namespace LeftMerge94626
def owner : Owner := ⟨.program ⟨257⟩, ⟨69298⟩⟩
def mergeEvent : Nat := 94626
def frameStart : Nat := 94531
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68560⟩⟩] } }
def rhsRaw : List Term := Proof.Events369.exact94573RawTerms
def group : MergeGroup := .relation 94625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94625) (rhsResult := 94573)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69295⟩⟩) ⟨68560⟩ 94573) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68560⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], [⟨.program ⟨257⟩, ⟨68560⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94626

namespace LeftMerge94634
def owner : Owner := ⟨.program ⟨257⟩, ⟨65830⟩⟩
def mergeEvent : Nat := 94634
def frameStart : Nat := 94531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94587RawTerms
def rightRaw : List Term := Proof.Events369.exact94630RawTerms
def group : MergeGroup := .operator 94587 94630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94587) (leftOrdinal := 0)
    (rightResult := 94630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94634

namespace LeftMerge94651
def owner : Owner := ⟨.program ⟨257⟩, ⟨67823⟩⟩
def mergeEvent : Nat := 94651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events369.exact94648RawTerms
def group : MergeGroup := .relation 94650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94650) (rhsResult := 94648)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67820⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 94649 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67820⟩⟩]⟩) (none) 94648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94651

namespace LeftMerge94652
def owner : Owner := ⟨.program ⟨257⟩, ⟨67823⟩⟩
def mergeEvent : Nat := 94652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩] } }
def rhsRaw : List Term := Proof.Events369.exact94648RawTerms
def group : MergeGroup := .relation 94650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94650) (rhsResult := 94648)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67820⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 94649 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67820⟩⟩]⟩) (none) 94648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94652

namespace LeftMerge94653
def owner : Owner := ⟨.program ⟨257⟩, ⟨67823⟩⟩
def mergeEvent : Nat := 94653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68560⟩⟩] } }
def rhsRaw : List Term := Proof.Events369.exact94648RawTerms
def group : MergeGroup := .relation 94650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94650) (rhsResult := 94648)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67820⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 94649 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67820⟩⟩]⟩) (none) 94648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68560⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], [⟨.program ⟨257⟩, ⟨68560⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94653

namespace LeftMerge94654
def owner : Owner := ⟨.program ⟨257⟩, ⟨67823⟩⟩
def mergeEvent : Nat := 94654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events369.exact94648RawTerms
def group : MergeGroup := .relation 94650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94650) (rhsResult := 94648)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67820⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 94649 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67820⟩⟩]⟩) (none) 94648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨65828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94654

namespace LeftMerge94659
def owner : Owner := ⟨.program ⟨257⟩, ⟨69297⟩⟩
def mergeEvent : Nat := 94659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68560⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94655RawTerms
def rightRaw : List Term := Proof.Events369.exact94469RawTerms
def group : MergeGroup := .operator 94655 94469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94655) (leftOrdinal := 2)
    (rightResult := 94469) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68560⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68560⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25790⟩⟩, ⟨.program ⟨257⟩, ⟨65580⟩⟩], [⟨.program ⟨257⟩, ⟨68560⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94659

namespace LeftMerge94660
def owner : Owner := ⟨.program ⟨257⟩, ⟨69297⟩⟩
def mergeEvent : Nat := 94660
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94655RawTerms
def rightRaw : List Term := Proof.Events369.exact94469RawTerms
def group : MergeGroup := .operator 94655 94469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94655) (leftOrdinal := 1)
    (rightResult := 94469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69295⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94660

namespace LeftMerge94668
def owner : Owner := ⟨.program ⟨257⟩, ⟨70574⟩⟩
def mergeEvent : Nat := 94668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70572⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94662RawTerms
def rightRaw : List Term := Proof.Events368.exact94385RawTerms
def group : MergeGroup := .operator 94662 94385
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94662) (leftOrdinal := 0)
    (rightResult := 94385) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70572⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70572⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94668

namespace LeftMerge94669
def owner : Owner := ⟨.program ⟨257⟩, ⟨70574⟩⟩
def mergeEvent : Nat := 94669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70572⟩⟩] } }
def leftRaw : List Term := Proof.Events369.exact94662RawTerms
def rightRaw : List Term := Proof.Events368.exact94385RawTerms
def group : MergeGroup := .operator 94662 94385
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94662) (leftOrdinal := 1)
    (rightResult := 94385) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70572⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨65828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70572⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94669

namespace LeftMerge94671
def owner : Owner := ⟨.program ⟨257⟩, ⟨70574⟩⟩
def mergeEvent : Nat := 94671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68727⟩⟩] } }
def rhsRaw : List Term := Proof.Events368.exact94382RawTerms
def group : MergeGroup := .relation 94670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94670) (rhsResult := 94382)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨65828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70572⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70572⟩⟩) ⟨68727⟩ 94382) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68727⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨65828⟩⟩], [⟨.program ⟨257⟩, ⟨68727⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94671

namespace LeftMerge94685
def owner : Owner := ⟨.program ⟨257⟩, ⟨68180⟩⟩
def mergeEvent : Nat := 94685
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68177⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events369.exact94679RawTerms
def group : MergeGroup := .operator 90620 94679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 94679) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68177⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68177⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94685

namespace LeftMerge94806
def owner : Owner := ⟨.program ⟨257⟩, ⟨69029⟩⟩
def mergeEvent : Nat := 94806
def frameStart : Nat := 94740
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events370.exact94802RawTerms
def rightRaw : List Term := Proof.Events370.exact94800RawTerms
def group : MergeGroup := .operator 94802 94800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94802) (leftOrdinal := 0)
    (rightResult := 94800) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94806

namespace LeftMerge94818
def owner : Owner := ⟨.program ⟨257⟩, ⟨70573⟩⟩
def mergeEvent : Nat := 94818
def frameStart : Nat := 94740
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70572⟩⟩] } }
def leftRaw : List Term := Proof.Events370.exact94814RawTerms
def rightRaw : List Term := Proof.Events370.exact94791RawTerms
def group : MergeGroup := .operator 94814 94791
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94814) (leftOrdinal := 0)
    (rightResult := 94791) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70572⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70572⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94818

namespace LeftMerge94819
def owner : Owner := ⟨.program ⟨257⟩, ⟨70573⟩⟩
def mergeEvent : Nat := 94819
def frameStart : Nat := 94740
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70572⟩⟩] } }
def leftRaw : List Term := Proof.Events370.exact94814RawTerms
def rightRaw : List Term := Proof.Events370.exact94791RawTerms
def group : MergeGroup := .operator 94814 94791
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94814) (leftOrdinal := 1)
    (rightResult := 94791) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70572⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70572⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94819

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
