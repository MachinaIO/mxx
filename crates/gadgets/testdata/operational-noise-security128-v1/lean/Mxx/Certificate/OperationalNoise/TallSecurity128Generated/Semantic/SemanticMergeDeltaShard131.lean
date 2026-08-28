import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge25869
def owner : Owner := ⟨.program ⟨257⟩, ⟨17265⟩⟩
def mergeEvent : Nat := 25869
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16797⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact25865RawTerms
def rightRaw : List Term := Proof.Events100.exact25679RawTerms
def group : MergeGroup := .operator 25865 25679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25865) (leftOrdinal := 2)
    (rightResult := 25679) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16797⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16797⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12251⟩⟩, ⟨.program ⟨257⟩, ⟨15266⟩⟩], [⟨.program ⟨257⟩, ⟨16797⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25869

namespace LeftMerge25870
def owner : Owner := ⟨.program ⟨257⟩, ⟨17265⟩⟩
def mergeEvent : Nat := 25870
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact25865RawTerms
def rightRaw : List Term := Proof.Events100.exact25679RawTerms
def group : MergeGroup := .operator 25865 25679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25865) (leftOrdinal := 1)
    (rightResult := 25679) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17263⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25870

namespace LeftMerge25878
def owner : Owner := ⟨.program ⟨257⟩, ⟨17519⟩⟩
def mergeEvent : Nat := 25878
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact25872RawTerms
def rightRaw : List Term := Proof.Events099.exact25576RawTerms
def group : MergeGroup := .operator 25872 25576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25872) (leftOrdinal := 1)
    (rightResult := 25576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17517⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25878

namespace LeftMerge25880
def owner : Owner := ⟨.program ⟨257⟩, ⟨17519⟩⟩
def mergeEvent : Nat := 25880
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16923⟩⟩] } }
def rhsRaw : List Term := Proof.Events099.exact25573RawTerms
def group : MergeGroup := .relation 25879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25879) (rhsResult := 25573)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17517⟩⟩) ⟨16923⟩ 25573) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16923⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25880

namespace LeftMerge25881
def owner : Owner := ⟨.program ⟨257⟩, ⟨17519⟩⟩
def mergeEvent : Nat := 25881
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact25872RawTerms
def rightRaw : List Term := Proof.Events099.exact25576RawTerms
def group : MergeGroup := .operator 25872 25576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25872) (leftOrdinal := 0)
    (rightResult := 25576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17517⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25881

namespace LeftMerge25895
def owner : Owner := ⟨.program ⟨257⟩, ⟨16425⟩⟩
def mergeEvent : Nat := 25895
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events101.exact25889RawTerms
def group : MergeGroup := .operator 17169 25889
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 25889) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16422⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25895

namespace LeftMerge26016
def owner : Owner := ⟨.program ⟨257⟩, ⟨17172⟩⟩
def mergeEvent : Nat := 26016
def frameStart : Nat := 25950
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact26012RawTerms
def rightRaw : List Term := Proof.Events101.exact26010RawTerms
def group : MergeGroup := .operator 26012 26010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26012) (leftOrdinal := 0)
    (rightResult := 26010) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26016

namespace LeftMerge26028
def owner : Owner := ⟨.program ⟨257⟩, ⟨17518⟩⟩
def mergeEvent : Nat := 26028
def frameStart : Nat := 25950
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact26024RawTerms
def rightRaw : List Term := Proof.Events101.exact26001RawTerms
def group : MergeGroup := .operator 26024 26001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26024) (leftOrdinal := 1)
    (rightResult := 26001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17517⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26028

namespace LeftMerge26030
def owner : Owner := ⟨.program ⟨257⟩, ⟨17518⟩⟩
def mergeEvent : Nat := 26030
def frameStart : Nat := 25950
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16923⟩⟩] } }
def rhsRaw : List Term := Proof.Events101.exact25998RawTerms
def group : MergeGroup := .relation 26029
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26029) (rhsResult := 25998)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17517⟩⟩) ⟨16923⟩ 25998) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16923⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26030

namespace LeftMerge26031
def owner : Owner := ⟨.program ⟨257⟩, ⟨17518⟩⟩
def mergeEvent : Nat := 26031
def frameStart : Nat := 25950
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact26024RawTerms
def rightRaw : List Term := Proof.Events101.exact26001RawTerms
def group : MergeGroup := .operator 26024 26001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26024) (leftOrdinal := 0)
    (rightResult := 26001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17517⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26031

namespace LeftMerge26039
def owner : Owner := ⟨.program ⟨257⟩, ⟨15896⟩⟩
def mergeEvent : Nat := 26039
def frameStart : Nat := 25950
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact26012RawTerms
def rightRaw : List Term := Proof.Events101.exact26035RawTerms
def group : MergeGroup := .operator 26012 26035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26012) (leftOrdinal := 0)
    (rightResult := 26035) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15895⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26039

namespace LeftMerge26056
def owner : Owner := ⟨.program ⟨257⟩, ⟨16425⟩⟩
def mergeEvent : Nat := 26056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16923⟩⟩] } }
def rhsRaw : List Term := Proof.Events101.exact26053RawTerms
def group : MergeGroup := .relation 26055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26055) (rhsResult := 26053)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 26054 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩) (none) 26053) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16923⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26056

namespace LeftMerge26057
def owner : Owner := ⟨.program ⟨257⟩, ⟨16425⟩⟩
def mergeEvent : Nat := 26057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩] } }
def rhsRaw : List Term := Proof.Events101.exact26053RawTerms
def group : MergeGroup := .relation 26055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26055) (rhsResult := 26053)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 26054 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩) (none) 26053) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17517⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26057

namespace LeftMerge26058
def owner : Owner := ⟨.program ⟨257⟩, ⟨16425⟩⟩
def mergeEvent : Nat := 26058
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events101.exact26053RawTerms
def group : MergeGroup := .relation 26055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26055) (rhsResult := 26053)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 26054 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩) (none) 26053) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26058

namespace LeftMerge26059
def owner : Owner := ⟨.program ⟨257⟩, ⟨16425⟩⟩
def mergeEvent : Nat := 26059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }
def rhsRaw : List Term := Proof.Events101.exact26053RawTerms
def group : MergeGroup := .relation 26055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 26055) (rhsResult := 26053)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 26054 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16422⟩⟩]⟩) (none) 26053) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge26059

namespace LeftMerge26064
def owner : Owner := ⟨.program ⟨257⟩, ⟨17520⟩⟩
def mergeEvent : Nat := 26064
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16923⟩⟩] } }
def leftRaw : List Term := Proof.Events101.exact26060RawTerms
def rightRaw : List Term := Proof.Events101.exact25882RawTerms
def group : MergeGroup := .operator 26060 25882
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 26060) (leftOrdinal := 2)
    (rightResult := 25882) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16923⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16923⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨15718⟩⟩], [⟨.program ⟨257⟩, ⟨16923⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge26064

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
