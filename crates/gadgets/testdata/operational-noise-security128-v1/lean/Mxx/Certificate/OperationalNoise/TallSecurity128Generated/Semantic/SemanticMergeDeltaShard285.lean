import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge49643
def owner : Owner := ⟨.program ⟨257⟩, ⟨29612⟩⟩
def mergeEvent : Nat := 49643
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events193.exact49637RawTerms
def group : MergeGroup := .operator 46745 49637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 49637) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29609⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49643

namespace LeftMerge49722
def owner : Owner := ⟨.program ⟨257⟩, ⟨28967⟩⟩
def mergeEvent : Nat := 49722
def frameStart : Nat := 49692
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events194.exact49718RawTerms
def rightRaw : List Term := Proof.Events194.exact49715RawTerms
def group : MergeGroup := .operator 49718 49715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49718) (leftOrdinal := 0)
    (rightResult := 49715) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13401⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49722

namespace LeftMerge49752
def owner : Owner := ⟨.program ⟨257⟩, ⟨30400⟩⟩
def mergeEvent : Nat := 49752
def frameStart : Nat := 49692
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49748RawTerms
def rightRaw : List Term := Proof.Events194.exact49746RawTerms
def group : MergeGroup := .operator 49748 49746
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49748) (leftOrdinal := 0)
    (rightResult := 49746) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49752

namespace LeftMerge49775
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def mergeEvent : Nat := 49775
def frameStart : Nat := 49692
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49771RawTerms
def rightRaw : List Term := Proof.Events194.exact49768RawTerms
def group : MergeGroup := .operator 49771 49768
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49771) (leftOrdinal := 0)
    (rightResult := 49768) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49775

namespace LeftMerge49784
def owner : Owner := ⟨.program ⟨257⟩, ⟨30690⟩⟩
def mergeEvent : Nat := 49784
def frameStart : Nat := 49692
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49780RawTerms
def rightRaw : List Term := Proof.Events194.exact49737RawTerms
def group : MergeGroup := .operator 49780 49737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49780) (leftOrdinal := 0)
    (rightResult := 49737) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30687⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49784

namespace LeftMerge49785
def owner : Owner := ⟨.program ⟨257⟩, ⟨30690⟩⟩
def mergeEvent : Nat := 49785
def frameStart : Nat := 49692
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49780RawTerms
def rightRaw : List Term := Proof.Events194.exact49737RawTerms
def group : MergeGroup := .operator 49780 49737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49780) (leftOrdinal := 1)
    (rightResult := 49737) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30687⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49785

namespace LeftMerge49787
def owner : Owner := ⟨.program ⟨257⟩, ⟨30690⟩⟩
def mergeEvent : Nat := 49787
def frameStart : Nat := 49692
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30137⟩⟩] } }
def rhsRaw : List Term := Proof.Events194.exact49734RawTerms
def group : MergeGroup := .relation 49786
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49786) (rhsResult := 49734)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30687⟩⟩) ⟨30137⟩ 49734) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30137⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], [⟨.program ⟨257⟩, ⟨30137⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49787

namespace LeftMerge49795
def owner : Owner := ⟨.program ⟨257⟩, ⟨29154⟩⟩
def mergeEvent : Nat := 49795
def frameStart : Nat := 49692
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49748RawTerms
def rightRaw : List Term := Proof.Events194.exact49791RawTerms
def group : MergeGroup := .operator 49748 49791
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49748) (leftOrdinal := 0)
    (rightResult := 49791) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49795

namespace LeftMerge49812
def owner : Owner := ⟨.program ⟨257⟩, ⟨29612⟩⟩
def mergeEvent : Nat := 49812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events194.exact49809RawTerms
def group : MergeGroup := .relation 49811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49811) (rhsResult := 49809)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49810 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩]⟩) (none) 49809) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49812

namespace LeftMerge49813
def owner : Owner := ⟨.program ⟨257⟩, ⟨29612⟩⟩
def mergeEvent : Nat := 49813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩] } }
def rhsRaw : List Term := Proof.Events194.exact49809RawTerms
def group : MergeGroup := .relation 49811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49811) (rhsResult := 49809)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49810 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩]⟩) (none) 49809) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49813

namespace LeftMerge49814
def owner : Owner := ⟨.program ⟨257⟩, ⟨29612⟩⟩
def mergeEvent : Nat := 49814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30137⟩⟩] } }
def rhsRaw : List Term := Proof.Events194.exact49809RawTerms
def group : MergeGroup := .relation 49811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49811) (rhsResult := 49809)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49810 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩]⟩) (none) 49809) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30137⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], [⟨.program ⟨257⟩, ⟨30137⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49814

namespace LeftMerge49815
def owner : Owner := ⟨.program ⟨257⟩, ⟨29612⟩⟩
def mergeEvent : Nat := 49815
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events194.exact49809RawTerms
def group : MergeGroup := .relation 49811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49811) (rhsResult := 49809)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49810 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29609⟩⟩]⟩) (none) 49809) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49815

namespace LeftMerge49820
def owner : Owner := ⟨.program ⟨257⟩, ⟨30689⟩⟩
def mergeEvent : Nat := 49820
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30137⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49816RawTerms
def rightRaw : List Term := Proof.Events193.exact49630RawTerms
def group : MergeGroup := .operator 49816 49630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49816) (leftOrdinal := 2)
    (rightResult := 49630) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30137⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30137⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨13401⟩⟩, ⟨.program ⟨257⟩, ⟨28966⟩⟩], [⟨.program ⟨257⟩, ⟨30137⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49820

namespace LeftMerge49821
def owner : Owner := ⟨.program ⟨257⟩, ⟨30689⟩⟩
def mergeEvent : Nat := 49821
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49816RawTerms
def rightRaw : List Term := Proof.Events193.exact49630RawTerms
def group : MergeGroup := .operator 49816 49630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49816) (leftOrdinal := 1)
    (rightResult := 49630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30687⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49821

namespace LeftMerge49829
def owner : Owner := ⟨.program ⟨257⟩, ⟨31171⟩⟩
def mergeEvent : Nat := 49829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31169⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49823RawTerms
def rightRaw : List Term := Proof.Events193.exact49546RawTerms
def group : MergeGroup := .operator 49823 49546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49823) (leftOrdinal := 0)
    (rightResult := 49546) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31169⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31169⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49829

namespace LeftMerge49830
def owner : Owner := ⟨.program ⟨257⟩, ⟨31171⟩⟩
def mergeEvent : Nat := 49830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31169⟩⟩] } }
def leftRaw : List Term := Proof.Events194.exact49823RawTerms
def rightRaw : List Term := Proof.Events193.exact49546RawTerms
def group : MergeGroup := .operator 49823 49546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49823) (leftOrdinal := 1)
    (rightResult := 49546) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31169⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨29152⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31169⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49830

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
