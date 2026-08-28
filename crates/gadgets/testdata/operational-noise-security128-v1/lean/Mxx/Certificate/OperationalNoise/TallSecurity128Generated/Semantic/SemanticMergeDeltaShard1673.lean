import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge271088
def owner : Owner := ⟨.program ⟨257⟩, ⟨61371⟩⟩
def mergeEvent : Nat := 271088
def frameStart : Nat := 270995
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩] } }
def leftRaw : List Term := Proof.Events1058.exact271083RawTerms
def rightRaw : List Term := Proof.Events1058.exact271040RawTerms
def group : MergeGroup := .operator 271083 271040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271083) (leftOrdinal := 1)
    (rightResult := 271040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61368⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271088

namespace LeftMerge271090
def owner : Owner := ⟨.program ⟨257⟩, ⟨61371⟩⟩
def mergeEvent : Nat := 271090
def frameStart : Nat := 270995
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60899⟩⟩] } }
def rhsRaw : List Term := Proof.Events1058.exact271037RawTerms
def group : MergeGroup := .relation 271089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271089) (rhsResult := 271037)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61368⟩⟩) ⟨60899⟩ 271037) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60899⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], [⟨.program ⟨257⟩, ⟨60899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271090

namespace LeftMerge271098
def owner : Owner := ⟨.program ⟨257⟩, ⟨59764⟩⟩
def mergeEvent : Nat := 271098
def frameStart : Nat := 270995
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1058.exact271051RawTerms
def rightRaw : List Term := Proof.Events1058.exact271094RawTerms
def group : MergeGroup := .operator 271051 271094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271051) (leftOrdinal := 0)
    (rightResult := 271094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271098

namespace LeftMerge271115
def owner : Owner := ⟨.program ⟨257⟩, ⟨60309⟩⟩
def mergeEvent : Nat := 271115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events1059.exact271112RawTerms
def group : MergeGroup := .relation 271114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271114) (rhsResult := 271112)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60306⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 271113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60306⟩⟩]⟩) (none) 271112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271115

namespace LeftMerge271116
def owner : Owner := ⟨.program ⟨257⟩, ⟨60309⟩⟩
def mergeEvent : Nat := 271116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩] } }
def rhsRaw : List Term := Proof.Events1059.exact271112RawTerms
def group : MergeGroup := .relation 271114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271114) (rhsResult := 271112)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60306⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 271113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60306⟩⟩]⟩) (none) 271112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271116

namespace LeftMerge271117
def owner : Owner := ⟨.program ⟨257⟩, ⟨60309⟩⟩
def mergeEvent : Nat := 271117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60899⟩⟩] } }
def rhsRaw : List Term := Proof.Events1059.exact271112RawTerms
def group : MergeGroup := .relation 271114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271114) (rhsResult := 271112)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60306⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 271113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60306⟩⟩]⟩) (none) 271112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60899⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], [⟨.program ⟨257⟩, ⟨60899⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271117

namespace LeftMerge271118
def owner : Owner := ⟨.program ⟨257⟩, ⟨60309⟩⟩
def mergeEvent : Nat := 271118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1059.exact271112RawTerms
def group : MergeGroup := .relation 271114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271114) (rhsResult := 271112)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60306⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 271113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60306⟩⟩]⟩) (none) 271112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271118

namespace LeftMerge271123
def owner : Owner := ⟨.program ⟨257⟩, ⟨61370⟩⟩
def mergeEvent : Nat := 271123
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60899⟩⟩] } }
def leftRaw : List Term := Proof.Events1059.exact271119RawTerms
def rightRaw : List Term := Proof.Events1058.exact270933RawTerms
def group : MergeGroup := .operator 271119 270933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271119) (leftOrdinal := 2)
    (rightResult := 270933) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60899⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60899⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨25150⟩⟩, ⟨.program ⟨257⟩, ⟨59260⟩⟩], [⟨.program ⟨257⟩, ⟨60899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271123

namespace LeftMerge271124
def owner : Owner := ⟨.program ⟨257⟩, ⟨61370⟩⟩
def mergeEvent : Nat := 271124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩] } }
def leftRaw : List Term := Proof.Events1059.exact271119RawTerms
def rightRaw : List Term := Proof.Events1058.exact270933RawTerms
def group : MergeGroup := .operator 271119 270933
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271119) (leftOrdinal := 1)
    (rightResult := 270933) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61368⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271124

namespace LeftMerge271132
def owner : Owner := ⟨.program ⟨257⟩, ⟨61637⟩⟩
def mergeEvent : Nat := 271132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61635⟩⟩] } }
def leftRaw : List Term := Proof.Events1059.exact271126RawTerms
def rightRaw : List Term := Proof.Events1058.exact270849RawTerms
def group : MergeGroup := .operator 271126 270849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271126) (leftOrdinal := 0)
    (rightResult := 270849) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61635⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61635⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271132

namespace LeftMerge271133
def owner : Owner := ⟨.program ⟨257⟩, ⟨61637⟩⟩
def mergeEvent : Nat := 271133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61635⟩⟩] } }
def leftRaw : List Term := Proof.Events1059.exact271126RawTerms
def rightRaw : List Term := Proof.Events1058.exact270849RawTerms
def group : MergeGroup := .operator 271126 270849
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271126) (leftOrdinal := 1)
    (rightResult := 270849) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61635⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61635⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271133

namespace LeftMerge271135
def owner : Owner := ⟨.program ⟨257⟩, ⟨61637⟩⟩
def mergeEvent : Nat := 271135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61026⟩⟩] } }
def rhsRaw : List Term := Proof.Events1057.exact270846RawTerms
def group : MergeGroup := .relation 271134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271134) (rhsResult := 270846)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61635⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61635⟩⟩) ⟨61026⟩ 270846) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61026⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨61026⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271135

namespace LeftMerge271149
def owner : Owner := ⟨.program ⟨257⟩, ⟨60533⟩⟩
def mergeEvent : Nat := 271149
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60530⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1059.exact271143RawTerms
def group : MergeGroup := .operator 266120 271143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 271143) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60530⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60530⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271149

namespace LeftMerge271270
def owner : Owner := ⟨.program ⟨257⟩, ⟨61276⟩⟩
def mergeEvent : Nat := 271270
def frameStart : Nat := 271204
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1059.exact271266RawTerms
def rightRaw : List Term := Proof.Events1059.exact271264RawTerms
def group : MergeGroup := .operator 271266 271264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271266) (leftOrdinal := 0)
    (rightResult := 271264) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271270

namespace LeftMerge271282
def owner : Owner := ⟨.program ⟨257⟩, ⟨61636⟩⟩
def mergeEvent : Nat := 271282
def frameStart : Nat := 271204
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61635⟩⟩] } }
def leftRaw : List Term := Proof.Events1059.exact271278RawTerms
def rightRaw : List Term := Proof.Events1059.exact271255RawTerms
def group : MergeGroup := .operator 271278 271255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271278) (leftOrdinal := 0)
    (rightResult := 271255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61635⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61635⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271282

namespace LeftMerge271283
def owner : Owner := ⟨.program ⟨257⟩, ⟨61636⟩⟩
def mergeEvent : Nat := 271283
def frameStart : Nat := 271204
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61635⟩⟩] } }
def leftRaw : List Term := Proof.Events1059.exact271278RawTerms
def rightRaw : List Term := Proof.Events1059.exact271255RawTerms
def group : MergeGroup := .operator 271278 271255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271278) (leftOrdinal := 1)
    (rightResult := 271255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61635⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61635⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271283

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
