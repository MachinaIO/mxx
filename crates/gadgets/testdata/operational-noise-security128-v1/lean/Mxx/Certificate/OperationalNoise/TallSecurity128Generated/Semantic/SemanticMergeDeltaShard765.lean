import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge126610
def owner : Owner := ⟨.program ⟨257⟩, ⟨33416⟩⟩
def mergeEvent : Nat := 126610
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩] } }
def leftRaw : List Term := Proof.Events494.exact126601RawTerms
def rightRaw : List Term := Proof.Events494.exact126537RawTerms
def group : MergeGroup := .operator 126601 126537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126601) (leftOrdinal := 0)
    (rightResult := 126537) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33415⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126610

namespace LeftMerge126624
def owner : Owner := ⟨.program ⟨257⟩, ⟨32352⟩⟩
def mergeEvent : Nat := 126624
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events494.exact126618RawTerms
def group : MergeGroup := .operator 119870 126618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 126618) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126624

namespace LeftMerge126703
def owner : Owner := ⟨.program ⟨257⟩, ⟨31378⟩⟩
def mergeEvent : Nat := 126703
def frameStart : Nat := 126673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events494.exact126699RawTerms
def rightRaw : List Term := Proof.Events494.exact126696RawTerms
def group : MergeGroup := .operator 126699 126696
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126699) (leftOrdinal := 0)
    (rightResult := 126696) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24242⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126703

namespace LeftMerge126733
def owner : Owner := ⟨.program ⟨257⟩, ⟨33212⟩⟩
def mergeEvent : Nat := 126733
def frameStart : Nat := 126673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events495.exact126729RawTerms
def rightRaw : List Term := Proof.Events495.exact126727RawTerms
def group : MergeGroup := .operator 126729 126727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126729) (leftOrdinal := 0)
    (rightResult := 126727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126733

namespace LeftMerge126756
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 126756
def frameStart : Nat := 126673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events495.exact126752RawTerms
def rightRaw : List Term := Proof.Events495.exact126749RawTerms
def group : MergeGroup := .operator 126752 126749
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126752) (leftOrdinal := 0)
    (rightResult := 126749) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126756

namespace LeftMerge126765
def owner : Owner := ⟨.program ⟨257⟩, ⟨33418⟩⟩
def mergeEvent : Nat := 126765
def frameStart : Nat := 126673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩] } }
def leftRaw : List Term := Proof.Events495.exact126761RawTerms
def rightRaw : List Term := Proof.Events494.exact126718RawTerms
def group : MergeGroup := .operator 126761 126718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126761) (leftOrdinal := 0)
    (rightResult := 126718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33415⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126765

namespace LeftMerge126766
def owner : Owner := ⟨.program ⟨257⟩, ⟨33418⟩⟩
def mergeEvent : Nat := 126766
def frameStart : Nat := 126673
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩] } }
def leftRaw : List Term := Proof.Events495.exact126761RawTerms
def rightRaw : List Term := Proof.Events494.exact126718RawTerms
def group : MergeGroup := .operator 126761 126718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126761) (leftOrdinal := 1)
    (rightResult := 126718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33415⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge126766

namespace LeftMerge126768
def owner : Owner := ⟨.program ⟨257⟩, ⟨33418⟩⟩
def mergeEvent : Nat := 126768
def frameStart : Nat := 126673
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32925⟩⟩] } }
def rhsRaw : List Term := Proof.Events494.exact126715RawTerms
def group : MergeGroup := .relation 126767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 126767) (rhsResult := 126715)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33415⟩⟩) ⟨32925⟩ 126715) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32925⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], [⟨.program ⟨257⟩, ⟨32925⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge126768

namespace LeftMerge126776
def owner : Owner := ⟨.program ⟨257⟩, ⟨31798⟩⟩
def mergeEvent : Nat := 126776
def frameStart : Nat := 126673
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events495.exact126729RawTerms
def rightRaw : List Term := Proof.Events495.exact126772RawTerms
def group : MergeGroup := .operator 126729 126772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126729) (leftOrdinal := 0)
    (rightResult := 126772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31796⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126776

namespace LeftMerge126793
def owner : Owner := ⟨.program ⟨257⟩, ⟨32352⟩⟩
def mergeEvent : Nat := 126793
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events495.exact126790RawTerms
def group : MergeGroup := .relation 126792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 126792) (rhsResult := 126790)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 126791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩]⟩) (none) 126790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126793

namespace LeftMerge126794
def owner : Owner := ⟨.program ⟨257⟩, ⟨32352⟩⟩
def mergeEvent : Nat := 126794
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩] } }
def rhsRaw : List Term := Proof.Events495.exact126790RawTerms
def group : MergeGroup := .relation 126792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 126792) (rhsResult := 126790)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 126791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩]⟩) (none) 126790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge126794

namespace LeftMerge126795
def owner : Owner := ⟨.program ⟨257⟩, ⟨32352⟩⟩
def mergeEvent : Nat := 126795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32925⟩⟩] } }
def rhsRaw : List Term := Proof.Events495.exact126790RawTerms
def group : MergeGroup := .relation 126792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 126792) (rhsResult := 126790)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 126791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩]⟩) (none) 126790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32925⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], [⟨.program ⟨257⟩, ⟨32925⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126795

namespace LeftMerge126796
def owner : Owner := ⟨.program ⟨257⟩, ⟨32352⟩⟩
def mergeEvent : Nat := 126796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨31796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events495.exact126790RawTerms
def group : MergeGroup := .relation 126792
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 126792) (rhsResult := 126790)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 126791 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32349⟩⟩]⟩) (none) 126790) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨31796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge126796

namespace LeftMerge126801
def owner : Owner := ⟨.program ⟨257⟩, ⟨33417⟩⟩
def mergeEvent : Nat := 126801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32925⟩⟩] } }
def leftRaw : List Term := Proof.Events495.exact126797RawTerms
def rightRaw : List Term := Proof.Events494.exact126611RawTerms
def group : MergeGroup := .operator 126797 126611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126797) (leftOrdinal := 2)
    (rightResult := 126611) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32925⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32925⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨24242⟩⟩, ⟨.program ⟨257⟩, ⟨31377⟩⟩], [⟨.program ⟨257⟩, ⟨32925⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge126801

namespace LeftMerge126802
def owner : Owner := ⟨.program ⟨257⟩, ⟨33417⟩⟩
def mergeEvent : Nat := 126802
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩] } }
def leftRaw : List Term := Proof.Events495.exact126797RawTerms
def rightRaw : List Term := Proof.Events494.exact126611RawTerms
def group : MergeGroup := .operator 126797 126611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126797) (leftOrdinal := 1)
    (rightResult := 126611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33415⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126802

namespace LeftMerge126810
def owner : Owner := ⟨.program ⟨257⟩, ⟨33770⟩⟩
def mergeEvent : Nat := 126810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33768⟩⟩] } }
def leftRaw : List Term := Proof.Events495.exact126804RawTerms
def rightRaw : List Term := Proof.Events494.exact126527RawTerms
def group : MergeGroup := .operator 126804 126527
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 126804) (leftOrdinal := 0)
    (rightResult := 126527) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33768⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge126810

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
