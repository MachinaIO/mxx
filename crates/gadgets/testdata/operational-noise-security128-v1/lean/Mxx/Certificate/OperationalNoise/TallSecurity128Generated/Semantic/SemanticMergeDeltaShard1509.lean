import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge245436
def owner : Owner := ⟨.program ⟨257⟩, ⟨16559⟩⟩
def mergeEvent : Nat := 245436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16983⟩⟩] } }
def rhsRaw : List Term := Proof.Events958.exact245431RawTerms
def group : MergeGroup := .relation 245433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245433) (rhsResult := 245431)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16556⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 245432 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16556⟩⟩]⟩) (none) 245431) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16983⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15772⟩⟩], [⟨.program ⟨257⟩, ⟨16983⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245436

namespace LeftMerge245437
def owner : Owner := ⟨.program ⟨257⟩, ⟨16559⟩⟩
def mergeEvent : Nat := 245437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨16003⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events958.exact245431RawTerms
def group : MergeGroup := .relation 245433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245433) (rhsResult := 245431)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16556⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 245432 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16556⟩⟩]⟩) (none) 245431) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16003⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨16003⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245437

namespace LeftMerge245442
def owner : Owner := ⟨.program ⟨257⟩, ⟨17708⟩⟩
def mergeEvent : Nat := 245442
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17705⟩⟩] } }
def leftRaw : List Term := Proof.Events958.exact245438RawTerms
def rightRaw : List Term := Proof.Events958.exact245260RawTerms
def group : MergeGroup := .operator 245438 245260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245438) (leftOrdinal := 0)
    (rightResult := 245260) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17705⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17705⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17705⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245442

namespace LeftMerge245443
def owner : Owner := ⟨.program ⟨257⟩, ⟨17708⟩⟩
def mergeEvent : Nat := 245443
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16983⟩⟩] } }
def leftRaw : List Term := Proof.Events958.exact245438RawTerms
def rightRaw : List Term := Proof.Events958.exact245260RawTerms
def group : MergeGroup := .operator 245438 245260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245438) (leftOrdinal := 2)
    (rightResult := 245260) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16983⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16983⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15772⟩⟩], [⟨.program ⟨257⟩, ⟨16983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245443

namespace LeftMerge245536
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 17)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245536

namespace LeftMerge245537
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48337⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 29)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48337⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245537

namespace LeftMerge245539
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48337⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236750RawTerms
def group : MergeGroup := .relation 245538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245538) (rhsResult := 236750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 236750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245539

namespace LeftMerge245540
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 16)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245540

namespace LeftMerge245541
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45657⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 28)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45657⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245541

namespace LeftMerge245543
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45657⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236750RawTerms
def group : MergeGroup := .relation 245542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245542) (rhsResult := 236750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 236750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245543

namespace LeftMerge245544
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 15)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245544

namespace LeftMerge245545
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42973⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 27)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42973⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245545

namespace LeftMerge245547
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42973⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236750RawTerms
def group : MergeGroup := .relation 245546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245546) (rhsResult := 236750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 236750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245547

namespace LeftMerge245548
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245548
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 14)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge245548

namespace LeftMerge245549
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245549
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events959.exact245530RawTerms
def rightRaw : List Term := Proof.Events924.exact236753RawTerms
def group : MergeGroup := .operator 245530 236753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 245530) (leftOrdinal := 26)
    (rightResult := 236753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245549

namespace LeftMerge245551
def owner : Owner := ⟨.program ⟨257⟩, ⟨71174⟩⟩
def mergeEvent : Nat := 245551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236750RawTerms
def group : MergeGroup := .relation 245550
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 245550) (rhsResult := 236750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 236750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40293⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge245551

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
