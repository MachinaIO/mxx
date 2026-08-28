import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge185809
def owner : Owner := ⟨.program ⟨257⟩, ⟨22739⟩⟩
def mergeEvent : Nat := 185809
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events725.exact185803RawTerms
def group : MergeGroup := .operator 178370 185803
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 185803) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22736⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185809

namespace LeftMerge185930
def owner : Owner := ⟨.program ⟨257⟩, ⟨23300⟩⟩
def mergeEvent : Nat := 185930
def frameStart : Nat := 185864
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact185926RawTerms
def rightRaw : List Term := Proof.Events726.exact185924RawTerms
def group : MergeGroup := .operator 185926 185924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185926) (leftOrdinal := 0)
    (rightResult := 185924) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185930

namespace LeftMerge185942
def owner : Owner := ⟨.program ⟨257⟩, ⟨23966⟩⟩
def mergeEvent : Nat := 185942
def frameStart : Nat := 185864
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact185938RawTerms
def rightRaw : List Term := Proof.Events726.exact185915RawTerms
def group : MergeGroup := .operator 185938 185915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185938) (leftOrdinal := 0)
    (rightResult := 185915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23965⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185942

namespace LeftMerge185943
def owner : Owner := ⟨.program ⟨257⟩, ⟨23966⟩⟩
def mergeEvent : Nat := 185943
def frameStart : Nat := 185864
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact185938RawTerms
def rightRaw : List Term := Proof.Events726.exact185915RawTerms
def group : MergeGroup := .operator 185938 185915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185938) (leftOrdinal := 1)
    (rightResult := 185915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23965⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185943

namespace LeftMerge185945
def owner : Owner := ⟨.program ⟨257⟩, ⟨23966⟩⟩
def mergeEvent : Nat := 185945
def frameStart : Nat := 185864
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23108⟩⟩] } }
def rhsRaw : List Term := Proof.Events726.exact185912RawTerms
def group : MergeGroup := .relation 185944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185944) (rhsResult := 185912)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23965⟩⟩) ⟨23108⟩ 185912) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23108⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨23108⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185945

namespace LeftMerge185953
def owner : Owner := ⟨.program ⟨257⟩, ⟨22145⟩⟩
def mergeEvent : Nat := 185953
def frameStart : Nat := 185864
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact185926RawTerms
def rightRaw : List Term := Proof.Events726.exact185949RawTerms
def group : MergeGroup := .operator 185926 185949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185926) (leftOrdinal := 0)
    (rightResult := 185949) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22143⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185953

namespace LeftMerge185970
def owner : Owner := ⟨.program ⟨257⟩, ⟨22739⟩⟩
def mergeEvent : Nat := 185970
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }
def rhsRaw : List Term := Proof.Events726.exact185967RawTerms
def group : MergeGroup := .relation 185969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185969) (rhsResult := 185967)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185968 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩]⟩) (none) 185967) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185970

namespace LeftMerge185971
def owner : Owner := ⟨.program ⟨257⟩, ⟨22739⟩⟩
def mergeEvent : Nat := 185971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩] } }
def rhsRaw : List Term := Proof.Events726.exact185967RawTerms
def group : MergeGroup := .relation 185969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185969) (rhsResult := 185967)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185968 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩]⟩) (none) 185967) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185971

namespace LeftMerge185972
def owner : Owner := ⟨.program ⟨257⟩, ⟨22739⟩⟩
def mergeEvent : Nat := 185972
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23108⟩⟩] } }
def rhsRaw : List Term := Proof.Events726.exact185967RawTerms
def group : MergeGroup := .relation 185969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185969) (rhsResult := 185967)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185968 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩]⟩) (none) 185967) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23108⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨23108⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185972

namespace LeftMerge185973
def owner : Owner := ⟨.program ⟨257⟩, ⟨22739⟩⟩
def mergeEvent : Nat := 185973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events726.exact185967RawTerms
def group : MergeGroup := .relation 185969
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 185969) (rhsResult := 185967)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 185968 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22736⟩⟩]⟩) (none) 185967) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22143⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨22143⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185973

namespace LeftMerge185978
def owner : Owner := ⟨.program ⟨257⟩, ⟨23968⟩⟩
def mergeEvent : Nat := 185978
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact185974RawTerms
def rightRaw : List Term := Proof.Events725.exact185796RawTerms
def group : MergeGroup := .operator 185974 185796
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185974) (leftOrdinal := 0)
    (rightResult := 185796) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23965⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge185978

namespace LeftMerge185979
def owner : Owner := ⟨.program ⟨257⟩, ⟨23968⟩⟩
def mergeEvent : Nat := 185979
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23108⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact185974RawTerms
def rightRaw : List Term := Proof.Events725.exact185796RawTerms
def group : MergeGroup := .operator 185974 185796
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 185974) (leftOrdinal := 2)
    (rightResult := 185796) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23108⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23108⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨21832⟩⟩], [⟨.program ⟨257⟩, ⟨23108⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge185979

namespace LeftMerge186005
def owner : Owner := ⟨.program ⟨257⟩, ⟨18349⟩⟩
def mergeEvent : Nat := 186005
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8690RawTerms
def rightRaw : List Term := Proof.Events696.exact178278RawTerms
def group : MergeGroup := .operator 8690 178278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8690) (leftOrdinal := 0)
    (rightResult := 178278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186005

namespace LeftMerge186010
def owner : Owner := ⟨.program ⟨257⟩, ⟨8953⟩⟩
def mergeEvent : Nat := 186010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178148RawTerms
def rightRaw : List Term := Proof.Events098.exact25096RawTerms
def group : MergeGroup := .operator 178148 25096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178148) (leftOrdinal := 0)
    (rightResult := 25096) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186010

namespace LeftMerge186027
def owner : Owner := ⟨.program ⟨257⟩, ⟨18352⟩⟩
def mergeEvent : Nat := 186027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact186021RawTerms
def rightRaw : List Term := Proof.Events033.exact8693RawTerms
def group : MergeGroup := .operator 186021 8693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186021) (leftOrdinal := 1)
    (rightResult := 8693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩, ⟨.program ⟨257⟩, ⟨18346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186027

namespace LeftMerge186028
def owner : Owner := ⟨.program ⟨257⟩, ⟨18352⟩⟩
def mergeEvent : Nat := 186028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def leftRaw : List Term := Proof.Events726.exact186021RawTerms
def rightRaw : List Term := Proof.Events033.exact8693RawTerms
def group : MergeGroup := .operator 186021 8693
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186021) (leftOrdinal := 0)
    (rightResult := 8693) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12726⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12726⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186028

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
