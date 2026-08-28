import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge24690
def owner : Owner := ⟨.program ⟨257⟩, ⟨22285⟩⟩
def mergeEvent : Nat := 24690
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events096.exact24684RawTerms
def group : MergeGroup := .operator 17169 24684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 24684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24690

namespace LeftMerge24769
def owner : Owner := ⟨.program ⟨257⟩, ⟨21287⟩⟩
def mergeEvent : Nat := 24769
def frameStart : Nat := 24739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events096.exact24765RawTerms
def rightRaw : List Term := Proof.Events096.exact24762RawTerms
def group : MergeGroup := .operator 24765 24762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24765) (leftOrdinal := 0)
    (rightResult := 24762) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20971⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24769

namespace LeftMerge24799
def owner : Owner := ⟨.program ⟨257⟩, ⟨23172⟩⟩
def mergeEvent : Nat := 24799
def frameStart : Nat := 24739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24795RawTerms
def rightRaw : List Term := Proof.Events096.exact24793RawTerms
def group : MergeGroup := .operator 24795 24793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24795) (leftOrdinal := 0)
    (rightResult := 24793) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24799

namespace LeftMerge24822
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 24822
def frameStart : Nat := 24739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24818RawTerms
def rightRaw : List Term := Proof.Events096.exact24815RawTerms
def group : MergeGroup := .operator 24818 24815
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24818) (leftOrdinal := 0)
    (rightResult := 24815) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24822

namespace LeftMerge24831
def owner : Owner := ⟨.program ⟨257⟩, ⟨23346⟩⟩
def mergeEvent : Nat := 24831
def frameStart : Nat := 24739
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24827RawTerms
def rightRaw : List Term := Proof.Events096.exact24784RawTerms
def group : MergeGroup := .operator 24827 24784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24827) (leftOrdinal := 1)
    (rightResult := 24784) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23343⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24831

namespace LeftMerge24833
def owner : Owner := ⟨.program ⟨257⟩, ⟨23346⟩⟩
def mergeEvent : Nat := 24833
def frameStart : Nat := 24739
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22877⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24781RawTerms
def group : MergeGroup := .relation 24832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24832) (rhsResult := 24781)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23343⟩⟩) ⟨22877⟩ 24781) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22877⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], [⟨.program ⟨257⟩, ⟨22877⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24833

namespace LeftMerge24834
def owner : Owner := ⟨.program ⟨257⟩, ⟨23346⟩⟩
def mergeEvent : Nat := 24834
def frameStart : Nat := 24739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24827RawTerms
def rightRaw : List Term := Proof.Events096.exact24784RawTerms
def group : MergeGroup := .operator 24827 24784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24827) (leftOrdinal := 0)
    (rightResult := 24784) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23343⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24834

namespace LeftMerge24842
def owner : Owner := ⟨.program ⟨257⟩, ⟨21740⟩⟩
def mergeEvent : Nat := 24842
def frameStart : Nat := 24739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24795RawTerms
def rightRaw : List Term := Proof.Events097.exact24838RawTerms
def group : MergeGroup := .operator 24795 24838
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24795) (leftOrdinal := 0)
    (rightResult := 24838) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21738⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24842

namespace LeftMerge24859
def owner : Owner := ⟨.program ⟨257⟩, ⟨22285⟩⟩
def mergeEvent : Nat := 24859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22877⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact24856RawTerms
def group : MergeGroup := .relation 24858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24858) (rhsResult := 24856)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24857 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩]⟩) (none) 24856) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22877⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], [⟨.program ⟨257⟩, ⟨22877⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24859

namespace LeftMerge24860
def owner : Owner := ⟨.program ⟨257⟩, ⟨22285⟩⟩
def mergeEvent : Nat := 24860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact24856RawTerms
def group : MergeGroup := .relation 24858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24858) (rhsResult := 24856)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24857 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩]⟩) (none) 24856) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24860

namespace LeftMerge24861
def owner : Owner := ⟨.program ⟨257⟩, ⟨22285⟩⟩
def mergeEvent : Nat := 24861
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact24856RawTerms
def group : MergeGroup := .relation 24858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24858) (rhsResult := 24856)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24857 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩]⟩) (none) 24856) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24861

namespace LeftMerge24862
def owner : Owner := ⟨.program ⟨257⟩, ⟨22285⟩⟩
def mergeEvent : Nat := 24862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events097.exact24856RawTerms
def group : MergeGroup := .relation 24858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24858) (rhsResult := 24856)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24857 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22282⟩⟩]⟩) (none) 24856) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24862

namespace LeftMerge24867
def owner : Owner := ⟨.program ⟨257⟩, ⟨23345⟩⟩
def mergeEvent : Nat := 24867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22877⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24863RawTerms
def rightRaw : List Term := Proof.Events096.exact24677RawTerms
def group : MergeGroup := .operator 24863 24677
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24863) (leftOrdinal := 2)
    (rightResult := 24677) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22877⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22877⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨20971⟩⟩, ⟨.program ⟨257⟩, ⟨21286⟩⟩], [⟨.program ⟨257⟩, ⟨22877⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24867

namespace LeftMerge24868
def owner : Owner := ⟨.program ⟨257⟩, ⟨23345⟩⟩
def mergeEvent : Nat := 24868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24863RawTerms
def rightRaw : List Term := Proof.Events096.exact24677RawTerms
def group : MergeGroup := .operator 24863 24677
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24863) (leftOrdinal := 1)
    (rightResult := 24677) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23343⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24868

namespace LeftMerge24876
def owner : Owner := ⟨.program ⟨257⟩, ⟨23604⟩⟩
def mergeEvent : Nat := 24876
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23602⟩⟩] } }
def leftRaw : List Term := Proof.Events097.exact24870RawTerms
def rightRaw : List Term := Proof.Events095.exact24574RawTerms
def group : MergeGroup := .operator 24870 24574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24870) (leftOrdinal := 1)
    (rightResult := 24574) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23602⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23602⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24876

namespace LeftMerge24878
def owner : Owner := ⟨.program ⟨257⟩, ⟨23604⟩⟩
def mergeEvent : Nat := 24878
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23003⟩⟩] } }
def rhsRaw : List Term := Proof.Events095.exact24571RawTerms
def group : MergeGroup := .relation 24877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24877) (rhsResult := 24571)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23602⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23602⟩⟩) ⟨23003⟩ 24571) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨21738⟩⟩], [⟨.program ⟨257⟩, ⟨23003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24878

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
