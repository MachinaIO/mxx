import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge237826
def owner : Owner := ⟨.program ⟨257⟩, ⟨44278⟩⟩
def mergeEvent : Nat := 237826
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩] } }
def leftRaw : List Term := Proof.Events928.exact237817RawTerms
def rightRaw : List Term := Proof.Events928.exact237753RawTerms
def group : MergeGroup := .operator 237817 237753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237817) (leftOrdinal := 0)
    (rightResult := 237753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44277⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237826

namespace LeftMerge237840
def owner : Owner := ⟨.program ⟨257⟩, ⟨43212⟩⟩
def mergeEvent : Nat := 237840
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events929.exact237834RawTerms
def group : MergeGroup := .operator 236870 237834
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 237834) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43209⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237840

namespace LeftMerge237919
def owner : Owner := ⟨.program ⟨257⟩, ⟨42427⟩⟩
def mergeEvent : Nat := 237919
def frameStart : Nat := 237889
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events929.exact237915RawTerms
def rightRaw : List Term := Proof.Events929.exact237912RawTerms
def group : MergeGroup := .operator 237915 237912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237915) (leftOrdinal := 0)
    (rightResult := 237912) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14451⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237919

namespace LeftMerge237949
def owner : Owner := ⟨.program ⟨257⟩, ⟨44060⟩⟩
def mergeEvent : Nat := 237949
def frameStart : Nat := 237889
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events929.exact237945RawTerms
def rightRaw : List Term := Proof.Events929.exact237943RawTerms
def group : MergeGroup := .operator 237945 237943
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237945) (leftOrdinal := 0)
    (rightResult := 237943) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237949

namespace LeftMerge237972
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def mergeEvent : Nat := 237972
def frameStart : Nat := 237889
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events929.exact237968RawTerms
def rightRaw : List Term := Proof.Events929.exact237965RawTerms
def group : MergeGroup := .operator 237968 237965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237968) (leftOrdinal := 0)
    (rightResult := 237965) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237972

namespace LeftMerge237981
def owner : Owner := ⟨.program ⟨257⟩, ⟨44280⟩⟩
def mergeEvent : Nat := 237981
def frameStart : Nat := 237889
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩] } }
def leftRaw : List Term := Proof.Events929.exact237977RawTerms
def rightRaw : List Term := Proof.Events929.exact237934RawTerms
def group : MergeGroup := .operator 237977 237934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237977) (leftOrdinal := 0)
    (rightResult := 237934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44277⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237981

namespace LeftMerge237982
def owner : Owner := ⟨.program ⟨257⟩, ⟨44280⟩⟩
def mergeEvent : Nat := 237982
def frameStart : Nat := 237889
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩] } }
def leftRaw : List Term := Proof.Events929.exact237977RawTerms
def rightRaw : List Term := Proof.Events929.exact237934RawTerms
def group : MergeGroup := .operator 237977 237934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237977) (leftOrdinal := 1)
    (rightResult := 237934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44277⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237982

namespace LeftMerge237984
def owner : Owner := ⟨.program ⟨257⟩, ⟨44280⟩⟩
def mergeEvent : Nat := 237984
def frameStart : Nat := 237889
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43777⟩⟩] } }
def rhsRaw : List Term := Proof.Events929.exact237931RawTerms
def group : MergeGroup := .relation 237983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237983) (rhsResult := 237931)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44277⟩⟩) ⟨43777⟩ 237931) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43777⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨43777⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237984

namespace LeftMerge237992
def owner : Owner := ⟨.program ⟨257⟩, ⟨42774⟩⟩
def mergeEvent : Nat := 237992
def frameStart : Nat := 237889
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events929.exact237945RawTerms
def rightRaw : List Term := Proof.Events929.exact237988RawTerms
def group : MergeGroup := .operator 237945 237988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237945) (leftOrdinal := 0)
    (rightResult := 237988) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42772⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237992

namespace LeftMerge238009
def owner : Owner := ⟨.program ⟨257⟩, ⟨43212⟩⟩
def mergeEvent : Nat := 238009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events929.exact238006RawTerms
def group : MergeGroup := .relation 238008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238008) (rhsResult := 238006)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238007 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩]⟩) (none) 238006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238009

namespace LeftMerge238010
def owner : Owner := ⟨.program ⟨257⟩, ⟨43212⟩⟩
def mergeEvent : Nat := 238010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩] } }
def rhsRaw : List Term := Proof.Events929.exact238006RawTerms
def group : MergeGroup := .relation 238008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238008) (rhsResult := 238006)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238007 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩]⟩) (none) 238006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238010

namespace LeftMerge238011
def owner : Owner := ⟨.program ⟨257⟩, ⟨43212⟩⟩
def mergeEvent : Nat := 238011
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43777⟩⟩] } }
def rhsRaw : List Term := Proof.Events929.exact238006RawTerms
def group : MergeGroup := .relation 238008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238008) (rhsResult := 238006)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238007 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩]⟩) (none) 238006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43777⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨43777⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238011

namespace LeftMerge238012
def owner : Owner := ⟨.program ⟨257⟩, ⟨43212⟩⟩
def mergeEvent : Nat := 238012
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events929.exact238006RawTerms
def group : MergeGroup := .relation 238008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238008) (rhsResult := 238006)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238007 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43209⟩⟩]⟩) (none) 238006) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨42772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238012

namespace LeftMerge238017
def owner : Owner := ⟨.program ⟨257⟩, ⟨44279⟩⟩
def mergeEvent : Nat := 238017
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43777⟩⟩] } }
def leftRaw : List Term := Proof.Events929.exact238013RawTerms
def rightRaw : List Term := Proof.Events929.exact237827RawTerms
def group : MergeGroup := .operator 238013 237827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238013) (leftOrdinal := 2)
    (rightResult := 237827) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43777⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43777⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14451⟩⟩, ⟨.program ⟨257⟩, ⟨42426⟩⟩], [⟨.program ⟨257⟩, ⟨43777⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238017

namespace LeftMerge238018
def owner : Owner := ⟨.program ⟨257⟩, ⟨44279⟩⟩
def mergeEvent : Nat := 238018
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩] } }
def leftRaw : List Term := Proof.Events929.exact238013RawTerms
def rightRaw : List Term := Proof.Events929.exact237827RawTerms
def group : MergeGroup := .operator 238013 237827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238013) (leftOrdinal := 1)
    (rightResult := 237827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44277⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238018

namespace LeftMerge238026
def owner : Owner := ⟨.program ⟨257⟩, ⟨44621⟩⟩
def mergeEvent : Nat := 238026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩] } }
def leftRaw : List Term := Proof.Events929.exact238020RawTerms
def rightRaw : List Term := Proof.Events928.exact237743RawTerms
def group : MergeGroup := .operator 238020 237743
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238020) (leftOrdinal := 0)
    (rightResult := 237743) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44619⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238026

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
