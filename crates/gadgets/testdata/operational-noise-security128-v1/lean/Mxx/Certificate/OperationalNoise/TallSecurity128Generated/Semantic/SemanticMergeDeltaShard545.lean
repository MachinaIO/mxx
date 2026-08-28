import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge90768
def owner : Owner := ⟨.program ⟨257⟩, ⟨49717⟩⟩
def mergeEvent : Nat := 90768
def frameStart : Nat := 90675
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩] } }
def leftRaw : List Term := Proof.Events354.exact90763RawTerms
def rightRaw : List Term := Proof.Events354.exact90720RawTerms
def group : MergeGroup := .operator 90763 90720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90763) (leftOrdinal := 1)
    (rightResult := 90720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49714⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90768

namespace LeftMerge90770
def owner : Owner := ⟨.program ⟨257⟩, ⟨49717⟩⟩
def mergeEvent : Nat := 90770
def frameStart : Nat := 90675
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49179⟩⟩] } }
def rhsRaw : List Term := Proof.Events354.exact90717RawTerms
def group : MergeGroup := .relation 90769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90769) (rhsResult := 90717)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49714⟩⟩) ⟨49179⟩ 90717) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨49179⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90770

namespace LeftMerge90778
def owner : Owner := ⟨.program ⟨257⟩, ⟨48190⟩⟩
def mergeEvent : Nat := 90778
def frameStart : Nat := 90675
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events354.exact90731RawTerms
def rightRaw : List Term := Proof.Events354.exact90774RawTerms
def group : MergeGroup := .operator 90731 90774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90731) (leftOrdinal := 0)
    (rightResult := 90774) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90778

namespace LeftMerge90795
def owner : Owner := ⟨.program ⟨257⟩, ⟨48642⟩⟩
def mergeEvent : Nat := 90795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events354.exact90792RawTerms
def group : MergeGroup := .relation 90794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90794) (rhsResult := 90792)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90793 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩) (none) 90792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90795

namespace LeftMerge90796
def owner : Owner := ⟨.program ⟨257⟩, ⟨48642⟩⟩
def mergeEvent : Nat := 90796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩] } }
def rhsRaw : List Term := Proof.Events354.exact90792RawTerms
def group : MergeGroup := .relation 90794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90794) (rhsResult := 90792)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90793 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩) (none) 90792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90796

namespace LeftMerge90797
def owner : Owner := ⟨.program ⟨257⟩, ⟨48642⟩⟩
def mergeEvent : Nat := 90797
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49179⟩⟩] } }
def rhsRaw : List Term := Proof.Events354.exact90792RawTerms
def group : MergeGroup := .relation 90794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90794) (rhsResult := 90792)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90793 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩) (none) 90792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨49179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90797

namespace LeftMerge90798
def owner : Owner := ⟨.program ⟨257⟩, ⟨48642⟩⟩
def mergeEvent : Nat := 90798
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events354.exact90792RawTerms
def group : MergeGroup := .relation 90794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90794) (rhsResult := 90792)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90793 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48639⟩⟩]⟩) (none) 90792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90798

namespace LeftMerge90803
def owner : Owner := ⟨.program ⟨257⟩, ⟨49716⟩⟩
def mergeEvent : Nat := 90803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49179⟩⟩] } }
def leftRaw : List Term := Proof.Events354.exact90799RawTerms
def rightRaw : List Term := Proof.Events353.exact90602RawTerms
def group : MergeGroup := .operator 90799 90602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90799) (leftOrdinal := 2)
    (rightResult := 90602) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49179⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49179⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15156⟩⟩, ⟨.program ⟨257⟩, ⟨47954⟩⟩], [⟨.program ⟨257⟩, ⟨49179⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90803

namespace LeftMerge90804
def owner : Owner := ⟨.program ⟨257⟩, ⟨49716⟩⟩
def mergeEvent : Nat := 90804
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩] } }
def leftRaw : List Term := Proof.Events354.exact90799RawTerms
def rightRaw : List Term := Proof.Events353.exact90602RawTerms
def group : MergeGroup := .operator 90799 90602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90799) (leftOrdinal := 1)
    (rightResult := 90602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49714⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90804

namespace LeftMerge90812
def owner : Owner := ⟨.program ⟨257⟩, ⟨50156⟩⟩
def mergeEvent : Nat := 90812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50154⟩⟩] } }
def leftRaw : List Term := Proof.Events354.exact90806RawTerms
def rightRaw : List Term := Proof.Events353.exact90513RawTerms
def group : MergeGroup := .operator 90806 90513
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90806) (leftOrdinal := 0)
    (rightResult := 90513) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50154⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50154⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90812

namespace LeftMerge90813
def owner : Owner := ⟨.program ⟨257⟩, ⟨50156⟩⟩
def mergeEvent : Nat := 90813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50154⟩⟩] } }
def leftRaw : List Term := Proof.Events354.exact90806RawTerms
def rightRaw : List Term := Proof.Events353.exact90513RawTerms
def group : MergeGroup := .operator 90806 90513
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90806) (leftOrdinal := 1)
    (rightResult := 90513) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50154⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50154⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90813

namespace LeftMerge90815
def owner : Owner := ⟨.program ⟨257⟩, ⟨50156⟩⟩
def mergeEvent : Nat := 90815
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49346⟩⟩] } }
def rhsRaw : List Term := Proof.Events353.exact90510RawTerms
def group : MergeGroup := .relation 90814
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90814) (rhsResult := 90510)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50154⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50154⟩⟩) ⟨49346⟩ 90510) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49346⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48188⟩⟩], [⟨.program ⟨257⟩, ⟨49346⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90815

namespace LeftMerge90829
def owner : Owner := ⟨.program ⟨257⟩, ⟨48999⟩⟩
def mergeEvent : Nat := 90829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48996⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events354.exact90823RawTerms
def group : MergeGroup := .operator 90620 90823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 90823) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48996⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48996⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90829

namespace LeftMerge90950
def owner : Owner := ⟨.program ⟨257⟩, ⟨49528⟩⟩
def mergeEvent : Nat := 90950
def frameStart : Nat := 90884
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events355.exact90946RawTerms
def rightRaw : List Term := Proof.Events355.exact90944RawTerms
def group : MergeGroup := .operator 90946 90944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90946) (leftOrdinal := 0)
    (rightResult := 90944) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90950

namespace LeftMerge90962
def owner : Owner := ⟨.program ⟨257⟩, ⟨50155⟩⟩
def mergeEvent : Nat := 90962
def frameStart : Nat := 90884
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50154⟩⟩] } }
def leftRaw : List Term := Proof.Events355.exact90958RawTerms
def rightRaw : List Term := Proof.Events355.exact90935RawTerms
def group : MergeGroup := .operator 90958 90935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90958) (leftOrdinal := 0)
    (rightResult := 90935) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50154⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50154⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90962

namespace LeftMerge90963
def owner : Owner := ⟨.program ⟨257⟩, ⟨50155⟩⟩
def mergeEvent : Nat := 90963
def frameStart : Nat := 90884
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50154⟩⟩] } }
def leftRaw : List Term := Proof.Events355.exact90958RawTerms
def rightRaw : List Term := Proof.Events355.exact90935RawTerms
def group : MergeGroup := .operator 90958 90935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90958) (leftOrdinal := 1)
    (rightResult := 90935) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48188⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50154⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48188⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50154⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90963

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
