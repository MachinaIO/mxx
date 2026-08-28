import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge149553
def owner : Owner := ⟨.program ⟨257⟩, ⟨14737⟩⟩
def mergeEvent : Nat := 149553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events026.exact6852RawTerms
def rightRaw : List Term := Proof.Events582.exact149028RawTerms
def group : MergeGroup := .operator 6852 149028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6852) (leftOrdinal := 0)
    (rightResult := 149028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14736⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149553

namespace LeftMerge149558
def owner : Owner := ⟨.program ⟨257⟩, ⟨8265⟩⟩
def mergeEvent : Nat := 149558
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148898RawTerms
def rightRaw : List Term := Proof.Events068.exact17622RawTerms
def group : MergeGroup := .operator 148898 17622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148898) (leftOrdinal := 0)
    (rightResult := 17622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149558

namespace LeftMerge149575
def owner : Owner := ⟨.program ⟨257⟩, ⟨14740⟩⟩
def mergeEvent : Nat := 149575
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events584.exact149569RawTerms
def rightRaw : List Term := Proof.Events068.exact17611RawTerms
def group : MergeGroup := .operator 149569 17611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149569) (leftOrdinal := 1)
    (rightResult := 17611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149575

namespace LeftMerge149577
def owner : Owner := ⟨.program ⟨257⟩, ⟨14740⟩⟩
def mergeEvent : Nat := 149577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def rhsRaw : List Term := Proof.Events068.exact17581RawTerms
def group : MergeGroup := .relation 149576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 149576) (rhsResult := 17581)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9562⟩⟩) ⟨7284⟩ 17581) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149577

namespace LeftMerge149578
def owner : Owner := ⟨.program ⟨257⟩, ⟨14740⟩⟩
def mergeEvent : Nat := 149578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events584.exact149569RawTerms
def rightRaw : List Term := Proof.Events068.exact17611RawTerms
def group : MergeGroup := .operator 149569 17611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149569) (leftOrdinal := 0)
    (rightResult := 17611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149578

namespace LeftMerge149583
def owner : Owner := ⟨.program ⟨257⟩, ⟨45089⟩⟩
def mergeEvent : Nat := 149583
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events584.exact149579RawTerms
def rightRaw : List Term := Proof.Events584.exact149549RawTerms
def group : MergeGroup := .operator 149579 149549
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149579) (leftOrdinal := 1)
    (rightResult := 149549) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149583

namespace LeftMerge149591
def owner : Owner := ⟨.program ⟨257⟩, ⟨46947⟩⟩
def mergeEvent : Nat := 149591
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩] } }
def leftRaw : List Term := Proof.Events584.exact149585RawTerms
def rightRaw : List Term := Proof.Events584.exact149521RawTerms
def group : MergeGroup := .operator 149585 149521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149585) (leftOrdinal := 1)
    (rightResult := 149521) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46946⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149591

namespace LeftMerge149593
def owner : Owner := ⟨.program ⟨257⟩, ⟨46947⟩⟩
def mergeEvent : Nat := 149593
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46451⟩⟩] } }
def rhsRaw : List Term := Proof.Events584.exact149518RawTerms
def group : MergeGroup := .relation 149592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 149592) (rhsResult := 149518)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46946⟩⟩) ⟨46451⟩ 149518) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46451⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], [⟨.program ⟨257⟩, ⟨46451⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149593

namespace LeftMerge149594
def owner : Owner := ⟨.program ⟨257⟩, ⟨46947⟩⟩
def mergeEvent : Nat := 149594
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩] } }
def leftRaw : List Term := Proof.Events584.exact149585RawTerms
def rightRaw : List Term := Proof.Events584.exact149521RawTerms
def group : MergeGroup := .operator 149585 149521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149585) (leftOrdinal := 0)
    (rightResult := 149521) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46946⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149594

namespace LeftMerge149608
def owner : Owner := ⟨.program ⟨257⟩, ⟨45882⟩⟩
def mergeEvent : Nat := 149608
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45879⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events584.exact149602RawTerms
def group : MergeGroup := .operator 149120 149602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 149602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45879⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149608

namespace LeftMerge149687
def owner : Owner := ⟨.program ⟨257⟩, ⟨45083⟩⟩
def mergeEvent : Nat := 149687
def frameStart : Nat := 149657
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events584.exact149683RawTerms
def rightRaw : List Term := Proof.Events584.exact149680RawTerms
def group : MergeGroup := .operator 149683 149680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149683) (leftOrdinal := 0)
    (rightResult := 149680) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14736⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149687

namespace LeftMerge149717
def owner : Owner := ⟨.program ⟨257⟩, ⟨46736⟩⟩
def mergeEvent : Nat := 149717
def frameStart : Nat := 149657
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events584.exact149713RawTerms
def rightRaw : List Term := Proof.Events584.exact149711RawTerms
def group : MergeGroup := .operator 149713 149711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149713) (leftOrdinal := 0)
    (rightResult := 149711) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149717

namespace LeftMerge149740
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 149740
def frameStart : Nat := 149657
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events584.exact149736RawTerms
def rightRaw : List Term := Proof.Events584.exact149733RawTerms
def group : MergeGroup := .operator 149736 149733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149736) (leftOrdinal := 0)
    (rightResult := 149733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149740

namespace LeftMerge149749
def owner : Owner := ⟨.program ⟨257⟩, ⟨46949⟩⟩
def mergeEvent : Nat := 149749
def frameStart : Nat := 149657
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩] } }
def leftRaw : List Term := Proof.Events584.exact149745RawTerms
def rightRaw : List Term := Proof.Events584.exact149702RawTerms
def group : MergeGroup := .operator 149745 149702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149745) (leftOrdinal := 0)
    (rightResult := 149702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46946⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149749

namespace LeftMerge149750
def owner : Owner := ⟨.program ⟨257⟩, ⟨46949⟩⟩
def mergeEvent : Nat := 149750
def frameStart : Nat := 149657
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩] } }
def leftRaw : List Term := Proof.Events584.exact149745RawTerms
def rightRaw : List Term := Proof.Events584.exact149702RawTerms
def group : MergeGroup := .operator 149745 149702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149745) (leftOrdinal := 1)
    (rightResult := 149702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46946⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149750

namespace LeftMerge149752
def owner : Owner := ⟨.program ⟨257⟩, ⟨46949⟩⟩
def mergeEvent : Nat := 149752
def frameStart : Nat := 149657
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46451⟩⟩] } }
def rhsRaw : List Term := Proof.Events584.exact149699RawTerms
def group : MergeGroup := .relation 149751
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 149751) (rhsResult := 149699)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46946⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46946⟩⟩) ⟨46451⟩ 149699) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46451⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14736⟩⟩, ⟨.program ⟨257⟩, ⟨45082⟩⟩], [⟨.program ⟨257⟩, ⟨46451⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149752

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
