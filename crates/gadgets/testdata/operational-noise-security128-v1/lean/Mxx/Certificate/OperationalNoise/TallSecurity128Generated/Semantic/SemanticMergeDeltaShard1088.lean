import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge178010
def owner : Owner := ⟨.program ⟨257⟩, ⟨71373⟩⟩
def mergeEvent : Nat := 178010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16377RawTerms
def group : MergeGroup := .relation 178009
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178009) (rhsResult := 16377)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9511⟩⟩) ⟨7255⟩ 16377) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32177⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178010

namespace LeftMerge178011
def owner : Owner := ⟨.program ⟨257⟩, ⟨71373⟩⟩
def mergeEvent : Nat := 178011
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact177957RawTerms
def rightRaw : List Term := Proof.Events064.exact16384RawTerms
def group : MergeGroup := .operator 177957 16384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177957) (leftOrdinal := 11)
    (rightResult := 16384) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9511⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178011

namespace LeftMerge178013
def owner : Owner := ⟨.program ⟨257⟩, ⟨71373⟩⟩
def mergeEvent : Nat := 178013
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16377RawTerms
def group : MergeGroup := .relation 178012
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178012) (rhsResult := 16377)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9511⟩⟩) ⟨7255⟩ 16377) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨22157⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178013

namespace LeftMerge178014
def owner : Owner := ⟨.program ⟨257⟩, ⟨71373⟩⟩
def mergeEvent : Nat := 178014
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact177957RawTerms
def rightRaw : List Term := Proof.Events064.exact16384RawTerms
def group : MergeGroup := .operator 177957 16384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177957) (leftOrdinal := 15)
    (rightResult := 16384) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9511⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178014

namespace LeftMerge178016
def owner : Owner := ⟨.program ⟨257⟩, ⟨71373⟩⟩
def mergeEvent : Nat := 178016
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16377RawTerms
def group : MergeGroup := .relation 178015
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178015) (rhsResult := 16377)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9511⟩⟩) ⟨7255⟩ 16377) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18937⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178016

namespace LeftMerge178017
def owner : Owner := ⟨.program ⟨257⟩, ⟨71373⟩⟩
def mergeEvent : Nat := 178017
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact177957RawTerms
def rightRaw : List Term := Proof.Events064.exact16384RawTerms
def group : MergeGroup := .operator 177957 16384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177957) (leftOrdinal := 18)
    (rightResult := 16384) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9511⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178017

namespace LeftMerge178019
def owner : Owner := ⟨.program ⟨257⟩, ⟨71373⟩⟩
def mergeEvent : Nat := 178019
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16377RawTerms
def group : MergeGroup := .relation 178018
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178018) (rhsResult := 16377)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9511⟩⟩) ⟨7255⟩ 16377) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨16094⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178019

namespace LeftMerge178020
def owner : Owner := ⟨.program ⟨257⟩, ⟨71373⟩⟩
def mergeEvent : Nat := 178020
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact177957RawTerms
def rightRaw : List Term := Proof.Events064.exact16384RawTerms
def group : MergeGroup := .operator 177957 16384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 177957) (leftOrdinal := 0)
    (rightResult := 16384) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9511⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7254⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9511⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178020

namespace LeftMerge178025
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def mergeEvent : Nat := 178025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178021RawTerms
def rightRaw : List Term := Proof.Events639.exact163618RawTerms
def group : MergeGroup := .operator 178021 163618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178021) (leftOrdinal := 6)
    (rightResult := 163618) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67538⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178025

namespace LeftMerge178026
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def mergeEvent : Nat := 178026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178021RawTerms
def rightRaw : List Term := Proof.Events639.exact163618RawTerms
def group : MergeGroup := .operator 178021 163618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178021) (leftOrdinal := 8)
    (rightResult := 163618) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48411⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178026

namespace LeftMerge178027
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def mergeEvent : Nat := 178027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178021RawTerms
def rightRaw : List Term := Proof.Events639.exact163618RawTerms
def group : MergeGroup := .operator 178021 163618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178021) (leftOrdinal := 9)
    (rightResult := 163618) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178027

namespace LeftMerge178028
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def mergeEvent : Nat := 178028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178021RawTerms
def rightRaw : List Term := Proof.Events639.exact163618RawTerms
def group : MergeGroup := .operator 178021 163618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178021) (leftOrdinal := 10)
    (rightResult := 163618) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨43054⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178028

namespace LeftMerge178029
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def mergeEvent : Nat := 178029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178021RawTerms
def rightRaw : List Term := Proof.Events639.exact163618RawTerms
def group : MergeGroup := .operator 178021 163618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178021) (leftOrdinal := 12)
    (rightResult := 163618) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40374⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178029

namespace LeftMerge178030
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def mergeEvent : Nat := 178030
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178021RawTerms
def rightRaw : List Term := Proof.Events639.exact163618RawTerms
def group : MergeGroup := .operator 178021 163618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178021) (leftOrdinal := 13)
    (rightResult := 163618) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37691⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178030

namespace LeftMerge178031
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def mergeEvent : Nat := 178031
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178021RawTerms
def rightRaw : List Term := Proof.Events639.exact163618RawTerms
def group : MergeGroup := .operator 178021 163618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178021) (leftOrdinal := 14)
    (rightResult := 163618) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35011⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178031

namespace LeftMerge178032
def owner : Owner := ⟨.program ⟨257⟩, ⟨71374⟩⟩
def mergeEvent : Nat := 178032
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178021RawTerms
def rightRaw : List Term := Proof.Events639.exact163618RawTerms
def group : MergeGroup := .operator 178021 163618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178021) (leftOrdinal := 16)
    (rightResult := 163618) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7255⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29354⟩⟩], [⟨.program ⟨257⟩, ⟨7255⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178032

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
