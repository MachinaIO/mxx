import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge136886
def owner : Owner := ⟨.program ⟨257⟩, ⟨34273⟩⟩
def mergeEvent : Nat := 136886
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events534.exact136882RawTerms
def rightRaw : List Term := Proof.Events534.exact136852RawTerms
def group : MergeGroup := .operator 136882 136852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136882) (leftOrdinal := 1)
    (rightResult := 136852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136886

namespace LeftMerge136894
def owner : Owner := ⟨.program ⟨257⟩, ⟨36183⟩⟩
def mergeEvent : Nat := 136894
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩] } }
def leftRaw : List Term := Proof.Events534.exact136888RawTerms
def rightRaw : List Term := Proof.Events534.exact136824RawTerms
def group : MergeGroup := .operator 136888 136824
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136888) (leftOrdinal := 1)
    (rightResult := 136824) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36182⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge136894

namespace LeftMerge136896
def owner : Owner := ⟨.program ⟨257⟩, ⟨36183⟩⟩
def mergeEvent : Nat := 136896
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35707⟩⟩] } }
def rhsRaw : List Term := Proof.Events534.exact136821RawTerms
def group : MergeGroup := .relation 136895
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 136895) (rhsResult := 136821)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36182⟩⟩) ⟨35707⟩ 136821) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35707⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], [⟨.program ⟨257⟩, ⟨35707⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge136896

namespace LeftMerge136897
def owner : Owner := ⟨.program ⟨257⟩, ⟨36183⟩⟩
def mergeEvent : Nat := 136897
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩] } }
def leftRaw : List Term := Proof.Events534.exact136888RawTerms
def rightRaw : List Term := Proof.Events534.exact136824RawTerms
def group : MergeGroup := .operator 136888 136824
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136888) (leftOrdinal := 0)
    (rightResult := 136824) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36182⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136897

namespace LeftMerge136911
def owner : Owner := ⟨.program ⟨257⟩, ⟨35122⟩⟩
def mergeEvent : Nat := 136911
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events534.exact136905RawTerms
def group : MergeGroup := .operator 134495 136905
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 136905) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35119⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136911

namespace LeftMerge136990
def owner : Owner := ⟨.program ⟨257⟩, ⟨34267⟩⟩
def mergeEvent : Nat := 136990
def frameStart : Nat := 136960
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events535.exact136986RawTerms
def rightRaw : List Term := Proof.Events535.exact136983RawTerms
def group : MergeGroup := .operator 136986 136983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 136986) (leftOrdinal := 0)
    (rightResult := 136983) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13476⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge136990

namespace LeftMerge137020
def owner : Owner := ⟨.program ⟨257⟩, ⟨36000⟩⟩
def mergeEvent : Nat := 137020
def frameStart : Nat := 136960
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events535.exact137016RawTerms
def rightRaw : List Term := Proof.Events535.exact137014RawTerms
def group : MergeGroup := .operator 137016 137014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137016) (leftOrdinal := 0)
    (rightResult := 137014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137020

namespace LeftMerge137043
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 137043
def frameStart : Nat := 136960
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events535.exact137039RawTerms
def rightRaw : List Term := Proof.Events535.exact137036RawTerms
def group : MergeGroup := .operator 137039 137036
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137039) (leftOrdinal := 0)
    (rightResult := 137036) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137043

namespace LeftMerge137052
def owner : Owner := ⟨.program ⟨257⟩, ⟨36185⟩⟩
def mergeEvent : Nat := 137052
def frameStart : Nat := 136960
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩] } }
def leftRaw : List Term := Proof.Events535.exact137048RawTerms
def rightRaw : List Term := Proof.Events535.exact137005RawTerms
def group : MergeGroup := .operator 137048 137005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137048) (leftOrdinal := 0)
    (rightResult := 137005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36182⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137052

namespace LeftMerge137053
def owner : Owner := ⟨.program ⟨257⟩, ⟨36185⟩⟩
def mergeEvent : Nat := 137053
def frameStart : Nat := 136960
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩] } }
def leftRaw : List Term := Proof.Events535.exact137048RawTerms
def rightRaw : List Term := Proof.Events535.exact137005RawTerms
def group : MergeGroup := .operator 137048 137005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137048) (leftOrdinal := 1)
    (rightResult := 137005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36182⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137053

namespace LeftMerge137055
def owner : Owner := ⟨.program ⟨257⟩, ⟨36185⟩⟩
def mergeEvent : Nat := 137055
def frameStart : Nat := 136960
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35707⟩⟩] } }
def rhsRaw : List Term := Proof.Events535.exact137002RawTerms
def group : MergeGroup := .relation 137054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137054) (rhsResult := 137002)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36182⟩⟩) ⟨35707⟩ 137002) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35707⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], [⟨.program ⟨257⟩, ⟨35707⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137055

namespace LeftMerge137063
def owner : Owner := ⟨.program ⟨257⟩, ⟨34694⟩⟩
def mergeEvent : Nat := 137063
def frameStart : Nat := 136960
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events535.exact137016RawTerms
def rightRaw : List Term := Proof.Events535.exact137059RawTerms
def group : MergeGroup := .operator 137016 137059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137016) (leftOrdinal := 0)
    (rightResult := 137059) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137063

namespace LeftMerge137080
def owner : Owner := ⟨.program ⟨257⟩, ⟨35122⟩⟩
def mergeEvent : Nat := 137080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events535.exact137077RawTerms
def group : MergeGroup := .relation 137079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137079) (rhsResult := 137077)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 137078 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩]⟩) (none) 137077) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137080

namespace LeftMerge137081
def owner : Owner := ⟨.program ⟨257⟩, ⟨35122⟩⟩
def mergeEvent : Nat := 137081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩] } }
def rhsRaw : List Term := Proof.Events535.exact137077RawTerms
def group : MergeGroup := .relation 137079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137079) (rhsResult := 137077)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 137078 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩]⟩) (none) 137077) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36182⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137081

namespace LeftMerge137082
def owner : Owner := ⟨.program ⟨257⟩, ⟨35122⟩⟩
def mergeEvent : Nat := 137082
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35707⟩⟩] } }
def rhsRaw : List Term := Proof.Events535.exact137077RawTerms
def group : MergeGroup := .relation 137079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137079) (rhsResult := 137077)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 137078 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩]⟩) (none) 137077) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35707⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13476⟩⟩, ⟨.program ⟨257⟩, ⟨34266⟩⟩], [⟨.program ⟨257⟩, ⟨35707⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137082

namespace LeftMerge137083
def owner : Owner := ⟨.program ⟨257⟩, ⟨35122⟩⟩
def mergeEvent : Nat := 137083
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events535.exact137077RawTerms
def group : MergeGroup := .relation 137079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137079) (rhsResult := 137077)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 137078 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35119⟩⟩]⟩) (none) 137077) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34692⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨34692⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137083

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
