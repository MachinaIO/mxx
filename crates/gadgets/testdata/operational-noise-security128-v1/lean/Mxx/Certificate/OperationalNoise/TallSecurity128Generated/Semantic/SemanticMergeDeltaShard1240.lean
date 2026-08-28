import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge203045
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203045
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37669⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 25)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37669⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203045

namespace LeftMerge203047
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203047
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37669⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events792.exact202849RawTerms
def group : MergeGroup := .relation 203046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203046) (rhsResult := 202849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 202849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37669⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203047

namespace LeftMerge203048
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203048
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 24)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203048

namespace LeftMerge203050
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203050
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events792.exact202849RawTerms
def group : MergeGroup := .relation 203049
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203049) (rhsResult := 202849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 202849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203050

namespace LeftMerge203051
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203051
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 22)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203051

namespace LeftMerge203053
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203053
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29325⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events792.exact202849RawTerms
def group : MergeGroup := .relation 203052
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203052) (rhsResult := 202849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 202849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29325⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203053

namespace LeftMerge203054
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203054
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26645⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 21)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26645⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203054

namespace LeftMerge203056
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203056
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26645⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events792.exact202849RawTerms
def group : MergeGroup := .relation 203055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203055) (rhsResult := 202849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 202849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26645⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203056

namespace LeftMerge203057
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203057
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 35)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203057

namespace LeftMerge203059
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203059
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events792.exact202849RawTerms
def group : MergeGroup := .relation 203058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203058) (rhsResult := 202849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 202849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203059

namespace LeftMerge203060
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203060
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 34)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203060

namespace LeftMerge203062
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203062
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63119⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events792.exact202849RawTerms
def group : MergeGroup := .relation 203061
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203061) (rhsResult := 202849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 202849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨63119⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203062

namespace LeftMerge203063
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203063
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60139⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 33)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60139⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203063

namespace LeftMerge203065
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203065
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60139⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events792.exact202849RawTerms
def group : MergeGroup := .relation 203064
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203064) (rhsResult := 202849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 202849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨60139⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203065

namespace LeftMerge203066
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203066
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩] } }
def leftRaw : List Term := Proof.Events793.exact203011RawTerms
def rightRaw : List Term := Proof.Events792.exact202852RawTerms
def group : MergeGroup := .operator 203011 202852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 203011) (leftOrdinal := 32)
    (rightResult := 202852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203066

namespace LeftMerge203068
def owner : Owner := ⟨.program ⟨257⟩, ⟨71298⟩⟩
def mergeEvent : Nat := 203068
def frameStart : Nat := 202336
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }
def rhsRaw : List Term := Proof.Events792.exact202849RawTerms
def group : MergeGroup := .relation 203067
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 203067) (rhsResult := 202849)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71297⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71297⟩⟩) ⟨68842⟩ 202849) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68842⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨68842⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge203068

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
