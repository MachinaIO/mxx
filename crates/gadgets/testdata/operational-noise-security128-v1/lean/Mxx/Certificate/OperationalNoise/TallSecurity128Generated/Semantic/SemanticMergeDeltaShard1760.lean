import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge284766
def owner : Owner := ⟨.program ⟨257⟩, ⟨69176⟩⟩
def mergeEvent : Nat := 284766
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68494⟩⟩] } }
def leftRaw : List Term := Proof.Events1112.exact284762RawTerms
def rightRaw : List Term := Proof.Events1111.exact284578RawTerms
def group : MergeGroup := .operator 284762 284578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284762) (leftOrdinal := 2)
    (rightResult := 284578) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68494⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68494⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25658⟩⟩, ⟨.program ⟨257⟩, ⟨65283⟩⟩], [⟨.program ⟨257⟩, ⟨68494⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284766

namespace LeftMerge284767
def owner : Owner := ⟨.program ⟨257⟩, ⟨69176⟩⟩
def mergeEvent : Nat := 284767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩] } }
def leftRaw : List Term := Proof.Events1112.exact284762RawTerms
def rightRaw : List Term := Proof.Events1111.exact284578RawTerms
def group : MergeGroup := .operator 284762 284578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284762) (leftOrdinal := 1)
    (rightResult := 284578) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69174⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284767

namespace LeftMerge284775
def owner : Owner := ⟨.program ⟨257⟩, ⟨69705⟩⟩
def mergeEvent : Nat := 284775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩] } }
def leftRaw : List Term := Proof.Events1112.exact284769RawTerms
def rightRaw : List Term := Proof.Events1111.exact284494RawTerms
def group : MergeGroup := .operator 284769 284494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284769) (leftOrdinal := 0)
    (rightResult := 284494) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69703⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284775

namespace LeftMerge284776
def owner : Owner := ⟨.program ⟨257⟩, ⟨69705⟩⟩
def mergeEvent : Nat := 284776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩] } }
def leftRaw : List Term := Proof.Events1112.exact284769RawTerms
def rightRaw : List Term := Proof.Events1111.exact284494RawTerms
def group : MergeGroup := .operator 284769 284494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284769) (leftOrdinal := 1)
    (rightResult := 284494) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69703⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284776

namespace LeftMerge284778
def owner : Owner := ⟨.program ⟨257⟩, ⟨69705⟩⟩
def mergeEvent : Nat := 284778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68628⟩⟩] } }
def rhsRaw : List Term := Proof.Events1111.exact284491RawTerms
def group : MergeGroup := .relation 284777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284777) (rhsResult := 284491)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69703⟩⟩) ⟨68628⟩ 284491) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68628⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨68628⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284778

namespace LeftMerge284792
def owner : Owner := ⟨.program ⟨257⟩, ⟨67960⟩⟩
def mergeEvent : Nat := 284792
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1112.exact284786RawTerms
def group : MergeGroup := .operator 280745 284786
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 284786) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67957⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284792

namespace LeftMerge284913
def owner : Owner := ⟨.program ⟨257⟩, ⟨68985⟩⟩
def mergeEvent : Nat := 284913
def frameStart : Nat := 284847
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1112.exact284909RawTerms
def rightRaw : List Term := Proof.Events1112.exact284907RawTerms
def group : MergeGroup := .operator 284909 284907
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284909) (leftOrdinal := 0)
    (rightResult := 284907) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284913

namespace LeftMerge284925
def owner : Owner := ⟨.program ⟨257⟩, ⟨69704⟩⟩
def mergeEvent : Nat := 284925
def frameStart : Nat := 284847
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩] } }
def leftRaw : List Term := Proof.Events1112.exact284921RawTerms
def rightRaw : List Term := Proof.Events1112.exact284898RawTerms
def group : MergeGroup := .operator 284921 284898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284921) (leftOrdinal := 0)
    (rightResult := 284898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69703⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284925

namespace LeftMerge284926
def owner : Owner := ⟨.program ⟨257⟩, ⟨69704⟩⟩
def mergeEvent : Nat := 284926
def frameStart : Nat := 284847
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩] } }
def leftRaw : List Term := Proof.Events1112.exact284921RawTerms
def rightRaw : List Term := Proof.Events1112.exact284898RawTerms
def group : MergeGroup := .operator 284921 284898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284921) (leftOrdinal := 1)
    (rightResult := 284898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69703⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284926

namespace LeftMerge284928
def owner : Owner := ⟨.program ⟨257⟩, ⟨69704⟩⟩
def mergeEvent : Nat := 284928
def frameStart : Nat := 284847
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68628⟩⟩] } }
def rhsRaw : List Term := Proof.Events1112.exact284895RawTerms
def group : MergeGroup := .relation 284927
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284927) (rhsResult := 284895)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69703⟩⟩) ⟨68628⟩ 284895) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68628⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨68628⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284928

namespace LeftMerge284936
def owner : Owner := ⟨.program ⟨257⟩, ⟨66192⟩⟩
def mergeEvent : Nat := 284936
def frameStart : Nat := 284847
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1112.exact284909RawTerms
def rightRaw : List Term := Proof.Events1113.exact284932RawTerms
def group : MergeGroup := .operator 284909 284932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284909) (leftOrdinal := 0)
    (rightResult := 284932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66181⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284936

namespace LeftMerge284953
def owner : Owner := ⟨.program ⟨257⟩, ⟨67960⟩⟩
def mergeEvent : Nat := 284953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }
def rhsRaw : List Term := Proof.Events1113.exact284950RawTerms
def group : MergeGroup := .relation 284952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284952) (rhsResult := 284950)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 284951 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩) (none) 284950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284953

namespace LeftMerge284954
def owner : Owner := ⟨.program ⟨257⟩, ⟨67960⟩⟩
def mergeEvent : Nat := 284954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩] } }
def rhsRaw : List Term := Proof.Events1113.exact284950RawTerms
def group : MergeGroup := .relation 284952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284952) (rhsResult := 284950)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 284951 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩) (none) 284950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284954

namespace LeftMerge284955
def owner : Owner := ⟨.program ⟨257⟩, ⟨67960⟩⟩
def mergeEvent : Nat := 284955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68628⟩⟩] } }
def rhsRaw : List Term := Proof.Events1113.exact284950RawTerms
def group : MergeGroup := .relation 284952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284952) (rhsResult := 284950)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 284951 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩) (none) 284950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65740⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68628⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨65740⟩⟩], [⟨.program ⟨257⟩, ⟨68628⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284955

namespace LeftMerge284956
def owner : Owner := ⟨.program ⟨257⟩, ⟨67960⟩⟩
def mergeEvent : Nat := 284956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1113.exact284950RawTerms
def group : MergeGroup := .relation 284952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 284952) (rhsResult := 284950)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 284951 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67957⟩⟩]⟩) (none) 284950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66181⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨66181⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge284956

namespace LeftMerge284961
def owner : Owner := ⟨.program ⟨257⟩, ⟨69706⟩⟩
def mergeEvent : Nat := 284961
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩] } }
def leftRaw : List Term := Proof.Events1113.exact284957RawTerms
def rightRaw : List Term := Proof.Events1112.exact284779RawTerms
def group : MergeGroup := .operator 284957 284779
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 284957) (leftOrdinal := 0)
    (rightResult := 284779) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge284961

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
