import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge97842
def owner : Owner := ⟨.program ⟨257⟩, ⟨23495⟩⟩
def mergeEvent : Nat := 97842
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩] } }
def leftRaw : List Term := Proof.Events382.exact97833RawTerms
def rightRaw : List Term := Proof.Events381.exact97769RawTerms
def group : MergeGroup := .operator 97833 97769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97833) (leftOrdinal := 0)
    (rightResult := 97769) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23494⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97842

namespace LeftMerge97856
def owner : Owner := ⟨.program ⟨257⟩, ⟨22422⟩⟩
def mergeEvent : Nat := 97856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events382.exact97850RawTerms
def group : MergeGroup := .operator 90620 97850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 97850) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22419⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97856

namespace LeftMerge97935
def owner : Owner := ⟨.program ⟨257⟩, ⟨21615⟩⟩
def mergeEvent : Nat := 97935
def frameStart : Nat := 97905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events382.exact97931RawTerms
def rightRaw : List Term := Proof.Events382.exact97928RawTerms
def group : MergeGroup := .operator 97931 97928
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97931) (leftOrdinal := 0)
    (rightResult := 97928) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21176⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97935

namespace LeftMerge97965
def owner : Owner := ⟨.program ⟨257⟩, ⟨23228⟩⟩
def mergeEvent : Nat := 97965
def frameStart : Nat := 97905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events382.exact97961RawTerms
def rightRaw : List Term := Proof.Events382.exact97959RawTerms
def group : MergeGroup := .operator 97961 97959
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97961) (leftOrdinal := 0)
    (rightResult := 97959) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97965

namespace LeftMerge97988
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 97988
def frameStart : Nat := 97905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events382.exact97984RawTerms
def rightRaw : List Term := Proof.Events382.exact97981RawTerms
def group : MergeGroup := .operator 97984 97981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97984) (leftOrdinal := 0)
    (rightResult := 97981) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97988

namespace LeftMerge97997
def owner : Owner := ⟨.program ⟨257⟩, ⟨23497⟩⟩
def mergeEvent : Nat := 97997
def frameStart : Nat := 97905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩] } }
def leftRaw : List Term := Proof.Events382.exact97993RawTerms
def rightRaw : List Term := Proof.Events382.exact97950RawTerms
def group : MergeGroup := .operator 97993 97950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97993) (leftOrdinal := 0)
    (rightResult := 97950) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23494⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97997

namespace LeftMerge97998
def owner : Owner := ⟨.program ⟨257⟩, ⟨23497⟩⟩
def mergeEvent : Nat := 97998
def frameStart : Nat := 97905
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩] } }
def leftRaw : List Term := Proof.Events382.exact97993RawTerms
def rightRaw : List Term := Proof.Events382.exact97950RawTerms
def group : MergeGroup := .operator 97993 97950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97993) (leftOrdinal := 1)
    (rightResult := 97950) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23494⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97998

namespace LeftMerge98000
def owner : Owner := ⟨.program ⟨257⟩, ⟨23497⟩⟩
def mergeEvent : Nat := 98000
def frameStart : Nat := 97905
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22959⟩⟩] } }
def rhsRaw : List Term := Proof.Events382.exact97947RawTerms
def group : MergeGroup := .relation 97999
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97999) (rhsResult := 97947)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23494⟩⟩) ⟨22959⟩ 97947) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], [⟨.program ⟨257⟩, ⟨22959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98000

namespace LeftMerge98008
def owner : Owner := ⟨.program ⟨257⟩, ⟨21850⟩⟩
def mergeEvent : Nat := 98008
def frameStart : Nat := 97905
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events382.exact97961RawTerms
def rightRaw : List Term := Proof.Events382.exact98004RawTerms
def group : MergeGroup := .operator 97961 98004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97961) (leftOrdinal := 0)
    (rightResult := 98004) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21848⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98008

namespace LeftMerge98025
def owner : Owner := ⟨.program ⟨257⟩, ⟨22422⟩⟩
def mergeEvent : Nat := 98025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events382.exact98022RawTerms
def group : MergeGroup := .relation 98024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98024) (rhsResult := 98022)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98023 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩]⟩) (none) 98022) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98025

namespace LeftMerge98026
def owner : Owner := ⟨.program ⟨257⟩, ⟨22422⟩⟩
def mergeEvent : Nat := 98026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩] } }
def rhsRaw : List Term := Proof.Events382.exact98022RawTerms
def group : MergeGroup := .relation 98024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98024) (rhsResult := 98022)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98023 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩]⟩) (none) 98022) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98026

namespace LeftMerge98027
def owner : Owner := ⟨.program ⟨257⟩, ⟨22422⟩⟩
def mergeEvent : Nat := 98027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22959⟩⟩] } }
def rhsRaw : List Term := Proof.Events382.exact98022RawTerms
def group : MergeGroup := .relation 98024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98024) (rhsResult := 98022)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98023 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩]⟩) (none) 98022) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], [⟨.program ⟨257⟩, ⟨22959⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98027

namespace LeftMerge98028
def owner : Owner := ⟨.program ⟨257⟩, ⟨22422⟩⟩
def mergeEvent : Nat := 98028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events382.exact98022RawTerms
def group : MergeGroup := .relation 98024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98024) (rhsResult := 98022)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98023 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22419⟩⟩]⟩) (none) 98022) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98028

namespace LeftMerge98033
def owner : Owner := ⟨.program ⟨257⟩, ⟨23496⟩⟩
def mergeEvent : Nat := 98033
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22959⟩⟩] } }
def leftRaw : List Term := Proof.Events382.exact98029RawTerms
def rightRaw : List Term := Proof.Events382.exact97843RawTerms
def group : MergeGroup := .operator 98029 97843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98029) (leftOrdinal := 2)
    (rightResult := 97843) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22959⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22959⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨21176⟩⟩, ⟨.program ⟨257⟩, ⟨21614⟩⟩], [⟨.program ⟨257⟩, ⟨22959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98033

namespace LeftMerge98034
def owner : Owner := ⟨.program ⟨257⟩, ⟨23496⟩⟩
def mergeEvent : Nat := 98034
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩] } }
def leftRaw : List Term := Proof.Events382.exact98029RawTerms
def rightRaw : List Term := Proof.Events382.exact97843RawTerms
def group : MergeGroup := .operator 98029 97843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98029) (leftOrdinal := 1)
    (rightResult := 97843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23494⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98034

namespace LeftMerge98042
def owner : Owner := ⟨.program ⟨257⟩, ⟨24029⟩⟩
def mergeEvent : Nat := 98042
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩] } }
def leftRaw : List Term := Proof.Events382.exact98036RawTerms
def rightRaw : List Term := Proof.Events381.exact97759RawTerms
def group : MergeGroup := .operator 98036 97759
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98036) (leftOrdinal := 0)
    (rightResult := 97759) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨24027⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24027⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98042

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
