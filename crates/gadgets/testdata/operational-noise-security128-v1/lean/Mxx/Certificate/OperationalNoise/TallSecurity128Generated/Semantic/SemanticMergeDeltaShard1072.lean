import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge175659
def owner : Owner := ⟨.program ⟨257⟩, ⟨70480⟩⟩
def mergeEvent : Nat := 175659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167787RawTerms
def rightRaw : List Term := Proof.Events686.exact175653RawTerms
def group : MergeGroup := .operator 167787 175653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167787) (leftOrdinal := 0)
    (rightResult := 175653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70478⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175659

namespace LeftMerge175660
def owner : Owner := ⟨.program ⟨257⟩, ⟨70480⟩⟩
def mergeEvent : Nat := 175660
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167787RawTerms
def rightRaw : List Term := Proof.Events686.exact175653RawTerms
def group : MergeGroup := .operator 167787 175653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167787) (leftOrdinal := 1)
    (rightResult := 175653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70478⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175660

namespace LeftMerge175662
def owner : Owner := ⟨.program ⟨257⟩, ⟨70480⟩⟩
def mergeEvent : Nat := 175662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68717⟩⟩] } }
def rhsRaw : List Term := Proof.Events686.exact175650RawTerms
def group : MergeGroup := .relation 175661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175661) (rhsResult := 175650)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70478⟩⟩) ⟨68717⟩ 175650) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68717⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨68717⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175662

namespace LeftMerge175676
def owner : Owner := ⟨.program ⟨257⟩, ⟨68156⟩⟩
def mergeEvent : Nat := 175676
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events686.exact175670RawTerms
def group : MergeGroup := .operator 163745 175670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 175670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175676

namespace LeftMerge175797
def owner : Owner := ⟨.program ⟨257⟩, ⟨69025⟩⟩
def mergeEvent : Nat := 175797
def frameStart : Nat := 175731
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events686.exact175793RawTerms
def rightRaw : List Term := Proof.Events686.exact175791RawTerms
def group : MergeGroup := .operator 175793 175791
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175793) (leftOrdinal := 0)
    (rightResult := 175791) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175797

namespace LeftMerge175809
def owner : Owner := ⟨.program ⟨257⟩, ⟨70479⟩⟩
def mergeEvent : Nat := 175809
def frameStart : Nat := 175731
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩] } }
def leftRaw : List Term := Proof.Events686.exact175805RawTerms
def rightRaw : List Term := Proof.Events686.exact175782RawTerms
def group : MergeGroup := .operator 175805 175782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175805) (leftOrdinal := 0)
    (rightResult := 175782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70478⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175809

namespace LeftMerge175810
def owner : Owner := ⟨.program ⟨257⟩, ⟨70479⟩⟩
def mergeEvent : Nat := 175810
def frameStart : Nat := 175731
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩] } }
def leftRaw : List Term := Proof.Events686.exact175805RawTerms
def rightRaw : List Term := Proof.Events686.exact175782RawTerms
def group : MergeGroup := .operator 175805 175782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175805) (leftOrdinal := 1)
    (rightResult := 175782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70478⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175810

namespace LeftMerge175812
def owner : Owner := ⟨.program ⟨257⟩, ⟨70479⟩⟩
def mergeEvent : Nat := 175812
def frameStart : Nat := 175731
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68717⟩⟩] } }
def rhsRaw : List Term := Proof.Events686.exact175779RawTerms
def group : MergeGroup := .relation 175811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175811) (rhsResult := 175779)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70478⟩⟩) ⟨68717⟩ 175779) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68717⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨68717⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175812

namespace LeftMerge175820
def owner : Owner := ⟨.program ⟨257⟩, ⟨66879⟩⟩
def mergeEvent : Nat := 175820
def frameStart : Nat := 175731
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events686.exact175793RawTerms
def rightRaw : List Term := Proof.Events686.exact175816RawTerms
def group : MergeGroup := .operator 175793 175816
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175793) (leftOrdinal := 0)
    (rightResult := 175816) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175820

namespace LeftMerge175837
def owner : Owner := ⟨.program ⟨257⟩, ⟨68156⟩⟩
def mergeEvent : Nat := 175837
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩] } }
def rhsRaw : List Term := Proof.Events686.exact175834RawTerms
def group : MergeGroup := .relation 175836
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175836) (rhsResult := 175834)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 175835 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩]⟩) (none) 175834) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175837

namespace LeftMerge175838
def owner : Owner := ⟨.program ⟨257⟩, ⟨68156⟩⟩
def mergeEvent : Nat := 175838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩] } }
def rhsRaw : List Term := Proof.Events686.exact175834RawTerms
def group : MergeGroup := .relation 175836
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175836) (rhsResult := 175834)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 175835 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩]⟩) (none) 175834) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175838

namespace LeftMerge175839
def owner : Owner := ⟨.program ⟨257⟩, ⟨68156⟩⟩
def mergeEvent : Nat := 175839
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68717⟩⟩] } }
def rhsRaw : List Term := Proof.Events686.exact175834RawTerms
def group : MergeGroup := .relation 175836
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175836) (rhsResult := 175834)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 175835 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩]⟩) (none) 175834) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68717⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨68717⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175839

namespace LeftMerge175840
def owner : Owner := ⟨.program ⟨257⟩, ⟨68156⟩⟩
def mergeEvent : Nat := 175840
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events686.exact175834RawTerms
def group : MergeGroup := .relation 175836
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 175836) (rhsResult := 175834)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 175835 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68153⟩⟩]⟩) (none) 175834) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨66868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175840

namespace LeftMerge175845
def owner : Owner := ⟨.program ⟨257⟩, ⟨70481⟩⟩
def mergeEvent : Nat := 175845
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩] } }
def leftRaw : List Term := Proof.Events686.exact175841RawTerms
def rightRaw : List Term := Proof.Events686.exact175663RawTerms
def group : MergeGroup := .operator 175841 175663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175841) (leftOrdinal := 0)
    (rightResult := 175663) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70478⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175845

namespace LeftMerge175846
def owner : Owner := ⟨.program ⟨257⟩, ⟨70481⟩⟩
def mergeEvent : Nat := 175846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68717⟩⟩] } }
def leftRaw : List Term := Proof.Events686.exact175841RawTerms
def rightRaw : List Term := Proof.Events686.exact175663RawTerms
def group : MergeGroup := .operator 175841 175663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175841) (leftOrdinal := 2)
    (rightResult := 175663) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68717⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68717⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨68717⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge175846

namespace LeftMerge175854
def owner : Owner := ⟨.program ⟨257⟩, ⟨70482⟩⟩
def mergeEvent : Nat := 175854
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩] } }
def leftRaw : List Term := Proof.Events686.exact175848RawTerms
def rightRaw : List Term := Proof.Events061.exact15702RawTerms
def group : MergeGroup := .operator 175848 15702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 175848) (leftOrdinal := 0)
    (rightResult := 15702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7215⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7173⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7215⟩⟩, ⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge175854

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
