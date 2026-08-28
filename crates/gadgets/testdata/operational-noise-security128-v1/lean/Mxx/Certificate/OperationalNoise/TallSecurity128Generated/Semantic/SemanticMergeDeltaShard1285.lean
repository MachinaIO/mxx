import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge208760
def owner : Owner := ⟨.program ⟨257⟩, ⟨43232⟩⟩
def mergeEvent : Nat := 208760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩] } }
def rhsRaw : List Term := Proof.Events815.exact208756RawTerms
def group : MergeGroup := .relation 208758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 208758) (rhsResult := 208756)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 208757 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43229⟩⟩]⟩) (none) 208756) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge208760

namespace LeftMerge208761
def owner : Owner := ⟨.program ⟨257⟩, ⟨43232⟩⟩
def mergeEvent : Nat := 208761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43789⟩⟩] } }
def rhsRaw : List Term := Proof.Events815.exact208756RawTerms
def group : MergeGroup := .relation 208758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 208758) (rhsResult := 208756)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 208757 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43229⟩⟩]⟩) (none) 208756) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43789⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], [⟨.program ⟨257⟩, ⟨43789⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge208761

namespace LeftMerge208762
def owner : Owner := ⟨.program ⟨257⟩, ⟨43232⟩⟩
def mergeEvent : Nat := 208762
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events815.exact208756RawTerms
def group : MergeGroup := .relation 208758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 208758) (rhsResult := 208756)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43229⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 208757 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43229⟩⟩]⟩) (none) 208756) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge208762

namespace LeftMerge208767
def owner : Owner := ⟨.program ⟨257⟩, ⟨44301⟩⟩
def mergeEvent : Nat := 208767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43789⟩⟩] } }
def leftRaw : List Term := Proof.Events815.exact208763RawTerms
def rightRaw : List Term := Proof.Events814.exact208577RawTerms
def group : MergeGroup := .operator 208763 208577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 208763) (leftOrdinal := 2)
    (rightResult := 208577) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43789⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43789⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨14481⟩⟩, ⟨.program ⟨257⟩, ⟨42474⟩⟩], [⟨.program ⟨257⟩, ⟨43789⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge208767

namespace LeftMerge208768
def owner : Owner := ⟨.program ⟨257⟩, ⟨44301⟩⟩
def mergeEvent : Nat := 208768
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩] } }
def leftRaw : List Term := Proof.Events815.exact208763RawTerms
def rightRaw : List Term := Proof.Events814.exact208577RawTerms
def group : MergeGroup := .operator 208763 208577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 208763) (leftOrdinal := 1)
    (rightResult := 208577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44299⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge208768

namespace LeftMerge208776
def owner : Owner := ⟨.program ⟨257⟩, ⟨44671⟩⟩
def mergeEvent : Nat := 208776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩] } }
def leftRaw : List Term := Proof.Events815.exact208770RawTerms
def rightRaw : List Term := Proof.Events814.exact208493RawTerms
def group : MergeGroup := .operator 208770 208493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 208770) (leftOrdinal := 0)
    (rightResult := 208493) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44669⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge208776

namespace LeftMerge208777
def owner : Owner := ⟨.program ⟨257⟩, ⟨44671⟩⟩
def mergeEvent : Nat := 208777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩] } }
def leftRaw : List Term := Proof.Events815.exact208770RawTerms
def rightRaw : List Term := Proof.Events814.exact208493RawTerms
def group : MergeGroup := .operator 208770 208493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 208770) (leftOrdinal := 1)
    (rightResult := 208493) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44669⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge208777

namespace LeftMerge208779
def owner : Owner := ⟨.program ⟨257⟩, ⟨44671⟩⟩
def mergeEvent : Nat := 208779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43941⟩⟩] } }
def rhsRaw : List Term := Proof.Events814.exact208490RawTerms
def group : MergeGroup := .relation 208778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 208778) (rhsResult := 208490)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44669⟩⟩) ⟨43941⟩ 208490) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43941⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨42788⟩⟩], [⟨.program ⟨257⟩, ⟨43941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge208779

namespace LeftMerge208793
def owner : Owner := ⟨.program ⟨257⟩, ⟨43539⟩⟩
def mergeEvent : Nat := 208793
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43536⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events815.exact208787RawTerms
def group : MergeGroup := .operator 207620 208787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 208787) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge208793

namespace LeftMerge208914
def owner : Owner := ⟨.program ⟨257⟩, ⟨44148⟩⟩
def mergeEvent : Nat := 208914
def frameStart : Nat := 208848
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events816.exact208910RawTerms
def rightRaw : List Term := Proof.Events816.exact208908RawTerms
def group : MergeGroup := .operator 208910 208908
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 208910) (leftOrdinal := 0)
    (rightResult := 208908) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge208914

namespace LeftMerge208926
def owner : Owner := ⟨.program ⟨257⟩, ⟨44670⟩⟩
def mergeEvent : Nat := 208926
def frameStart : Nat := 208848
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩] } }
def leftRaw : List Term := Proof.Events816.exact208922RawTerms
def rightRaw : List Term := Proof.Events816.exact208899RawTerms
def group : MergeGroup := .operator 208922 208899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 208922) (leftOrdinal := 0)
    (rightResult := 208899) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44669⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge208926

namespace LeftMerge208927
def owner : Owner := ⟨.program ⟨257⟩, ⟨44670⟩⟩
def mergeEvent : Nat := 208927
def frameStart : Nat := 208848
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩] } }
def leftRaw : List Term := Proof.Events816.exact208922RawTerms
def rightRaw : List Term := Proof.Events816.exact208899RawTerms
def group : MergeGroup := .operator 208922 208899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 208922) (leftOrdinal := 1)
    (rightResult := 208899) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44669⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge208927

namespace LeftMerge208929
def owner : Owner := ⟨.program ⟨257⟩, ⟨44670⟩⟩
def mergeEvent : Nat := 208929
def frameStart : Nat := 208848
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43941⟩⟩] } }
def rhsRaw : List Term := Proof.Events816.exact208896RawTerms
def group : MergeGroup := .relation 208928
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 208928) (rhsResult := 208896)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44669⟩⟩) ⟨43941⟩ 208896) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43941⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42788⟩⟩], [⟨.program ⟨257⟩, ⟨43941⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge208929

namespace LeftMerge208937
def owner : Owner := ⟨.program ⟨257⟩, ⟨43000⟩⟩
def mergeEvent : Nat := 208937
def frameStart : Nat := 208848
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events816.exact208910RawTerms
def rightRaw : List Term := Proof.Events816.exact208933RawTerms
def group : MergeGroup := .operator 208910 208933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 208910) (leftOrdinal := 0)
    (rightResult := 208933) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge208937

namespace LeftMerge208954
def owner : Owner := ⟨.program ⟨257⟩, ⟨43539⟩⟩
def mergeEvent : Nat := 208954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }
def rhsRaw : List Term := Proof.Events816.exact208951RawTerms
def group : MergeGroup := .relation 208953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 208953) (rhsResult := 208951)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 208952 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43536⟩⟩]⟩) (none) 208951) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge208954

namespace LeftMerge208955
def owner : Owner := ⟨.program ⟨257⟩, ⟨43539⟩⟩
def mergeEvent : Nat := 208955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩] } }
def rhsRaw : List Term := Proof.Events816.exact208951RawTerms
def group : MergeGroup := .relation 208953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 208953) (rhsResult := 208951)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 208952 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43536⟩⟩]⟩) (none) 208951) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44669⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge208955

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
