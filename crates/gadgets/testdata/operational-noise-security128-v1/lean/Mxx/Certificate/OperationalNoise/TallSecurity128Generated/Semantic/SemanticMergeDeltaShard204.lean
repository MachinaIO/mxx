import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge37088
def owner : Owner := ⟨.program ⟨257⟩, ⟨61561⟩⟩
def mergeEvent : Nat := 37088
def frameStart : Nat := 36995
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37083RawTerms
def rightRaw : List Term := Proof.Events144.exact37040RawTerms
def group : MergeGroup := .operator 37083 37040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37083) (leftOrdinal := 1)
    (rightResult := 37040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61558⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37088

namespace LeftMerge37090
def owner : Owner := ⟨.program ⟨257⟩, ⟨61561⟩⟩
def mergeEvent : Nat := 37090
def frameStart : Nat := 36995
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61003⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact37037RawTerms
def group : MergeGroup := .relation 37089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37089) (rhsResult := 37037)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61558⟩⟩) ⟨61003⟩ 37037) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], [⟨.program ⟨257⟩, ⟨61003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37090

namespace LeftMerge37098
def owner : Owner := ⟨.program ⟨257⟩, ⟨59902⟩⟩
def mergeEvent : Nat := 37098
def frameStart : Nat := 36995
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37051RawTerms
def rightRaw : List Term := Proof.Events144.exact37094RawTerms
def group : MergeGroup := .operator 37051 37094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37051) (leftOrdinal := 0)
    (rightResult := 37094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37098

namespace LeftMerge37115
def owner : Owner := ⟨.program ⟨257⟩, ⟨60482⟩⟩
def mergeEvent : Nat := 37115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact37112RawTerms
def group : MergeGroup := .relation 37114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37114) (rhsResult := 37112)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60479⟩⟩]⟩) (none) 37112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37115

namespace LeftMerge37116
def owner : Owner := ⟨.program ⟨257⟩, ⟨60482⟩⟩
def mergeEvent : Nat := 37116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact37112RawTerms
def group : MergeGroup := .relation 37114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37114) (rhsResult := 37112)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60479⟩⟩]⟩) (none) 37112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37116

namespace LeftMerge37117
def owner : Owner := ⟨.program ⟨257⟩, ⟨60482⟩⟩
def mergeEvent : Nat := 37117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61003⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact37112RawTerms
def group : MergeGroup := .relation 37114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37114) (rhsResult := 37112)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60479⟩⟩]⟩) (none) 37112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], [⟨.program ⟨257⟩, ⟨61003⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37117

namespace LeftMerge37118
def owner : Owner := ⟨.program ⟨257⟩, ⟨60482⟩⟩
def mergeEvent : Nat := 37118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events144.exact37112RawTerms
def group : MergeGroup := .relation 37114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37114) (rhsResult := 37112)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60479⟩⟩]⟩) (none) 37112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨59900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37118

namespace LeftMerge37123
def owner : Owner := ⟨.program ⟨257⟩, ⟨61560⟩⟩
def mergeEvent : Nat := 37123
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61003⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37119RawTerms
def rightRaw : List Term := Proof.Events144.exact36933RawTerms
def group : MergeGroup := .operator 37119 36933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37119) (leftOrdinal := 2)
    (rightResult := 36933) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61003⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61003⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25358⟩⟩, ⟨.program ⟨257⟩, ⟨59728⟩⟩], [⟨.program ⟨257⟩, ⟨61003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37123

namespace LeftMerge37124
def owner : Owner := ⟨.program ⟨257⟩, ⟨61560⟩⟩
def mergeEvent : Nat := 37124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩] } }
def leftRaw : List Term := Proof.Events144.exact37119RawTerms
def rightRaw : List Term := Proof.Events144.exact36933RawTerms
def group : MergeGroup := .operator 37119 36933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37119) (leftOrdinal := 1)
    (rightResult := 36933) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61558⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37124

namespace LeftMerge37132
def owner : Owner := ⟨.program ⟨257⟩, ⟨62173⟩⟩
def mergeEvent : Nat := 37132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62171⟩⟩] } }
def leftRaw : List Term := Proof.Events145.exact37126RawTerms
def rightRaw : List Term := Proof.Events143.exact36849RawTerms
def group : MergeGroup := .operator 37126 36849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37126) (leftOrdinal := 0)
    (rightResult := 36849) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62171⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62171⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37132

namespace LeftMerge37133
def owner : Owner := ⟨.program ⟨257⟩, ⟨62173⟩⟩
def mergeEvent : Nat := 37133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62171⟩⟩] } }
def leftRaw : List Term := Proof.Events145.exact37126RawTerms
def rightRaw : List Term := Proof.Events143.exact36849RawTerms
def group : MergeGroup := .operator 37126 36849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37126) (leftOrdinal := 1)
    (rightResult := 36849) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62171⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨59900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62171⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37133

namespace LeftMerge37135
def owner : Owner := ⟨.program ⟨257⟩, ⟨62173⟩⟩
def mergeEvent : Nat := 37135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61182⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36846RawTerms
def group : MergeGroup := .relation 37134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37134) (rhsResult := 36846)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨59900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62171⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨62171⟩⟩) ⟨61182⟩ 36846) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨59900⟩⟩], [⟨.program ⟨257⟩, ⟨61182⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37135

namespace LeftMerge37149
def owner : Owner := ⟨.program ⟨257⟩, ⟨60879⟩⟩
def mergeEvent : Nat := 37149
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60876⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events145.exact37143RawTerms
def group : MergeGroup := .operator 32120 37143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 37143) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60876⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37149

namespace LeftMerge37270
def owner : Owner := ⟨.program ⟨257⟩, ⟨61344⟩⟩
def mergeEvent : Nat := 37270
def frameStart : Nat := 37204
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events145.exact37266RawTerms
def rightRaw : List Term := Proof.Events145.exact37264RawTerms
def group : MergeGroup := .operator 37266 37264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37266) (leftOrdinal := 0)
    (rightResult := 37264) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37270

namespace LeftMerge37282
def owner : Owner := ⟨.program ⟨257⟩, ⟨62172⟩⟩
def mergeEvent : Nat := 37282
def frameStart : Nat := 37204
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62171⟩⟩] } }
def leftRaw : List Term := Proof.Events145.exact37278RawTerms
def rightRaw : List Term := Proof.Events145.exact37255RawTerms
def group : MergeGroup := .operator 37278 37255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37278) (leftOrdinal := 0)
    (rightResult := 37255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62171⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62171⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37282

namespace LeftMerge37283
def owner : Owner := ⟨.program ⟨257⟩, ⟨62172⟩⟩
def mergeEvent : Nat := 37283
def frameStart : Nat := 37204
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62171⟩⟩] } }
def leftRaw : List Term := Proof.Events145.exact37278RawTerms
def rightRaw : List Term := Proof.Events145.exact37255RawTerms
def group : MergeGroup := .operator 37278 37255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37278) (leftOrdinal := 1)
    (rightResult := 37255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59900⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62171⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59900⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62171⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37283

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
