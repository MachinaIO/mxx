import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge159181
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159181
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def rhsRaw : List Term := Proof.Events620.exact158974RawTerms
def group : MergeGroup := .relation 159180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 159180) (rhsResult := 158974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 158974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26580⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159181

namespace LeftMerge159182
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159182
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 35)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159182

namespace LeftMerge159184
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159184
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66391⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def rhsRaw : List Term := Proof.Events620.exact158974RawTerms
def group : MergeGroup := .relation 159183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 159183) (rhsResult := 158974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 158974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨66391⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159184

namespace LeftMerge159185
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159185
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 34)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159185

namespace LeftMerge159187
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159187
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def rhsRaw : List Term := Proof.Events620.exact158974RawTerms
def group : MergeGroup := .relation 159186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 159186) (rhsResult := 158974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 158974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨63024⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159187

namespace LeftMerge159188
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159188
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 33)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159188

namespace LeftMerge159190
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159190
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60044⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def rhsRaw : List Term := Proof.Events620.exact158974RawTerms
def group : MergeGroup := .relation 159189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 159189) (rhsResult := 158974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 158974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨60044⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159190

namespace LeftMerge159191
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159191
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 32)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159191

namespace LeftMerge159193
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159193
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57064⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def rhsRaw : List Term := Proof.Events620.exact158974RawTerms
def group : MergeGroup := .relation 159192
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 159192) (rhsResult := 158974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 158974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨57064⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159193

namespace LeftMerge159194
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159194
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 31)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159194

namespace LeftMerge159196
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159196
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54084⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def rhsRaw : List Term := Proof.Events620.exact158974RawTerms
def group : MergeGroup := .relation 159195
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 159195) (rhsResult := 158974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 158974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨54084⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159196

namespace LeftMerge159197
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159197
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 30)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159197

namespace LeftMerge159199
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159199
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51104⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def rhsRaw : List Term := Proof.Events620.exact158974RawTerms
def group : MergeGroup := .relation 159198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 159198) (rhsResult := 158974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 158974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨51104⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159199

namespace LeftMerge159200
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159200
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 23)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159200

namespace LeftMerge159202
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159202
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32049⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }
def rhsRaw : List Term := Proof.Events620.exact158974RawTerms
def group : MergeGroup := .relation 159201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 159201) (rhsResult := 158974)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71142⟩⟩) ⟨68812⟩ 158974) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68812⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨32049⟩⟩], [⟨.program ⟨257⟩, ⟨68812⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159202

namespace LeftMerge159203
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def mergeEvent : Nat := 159203
def frameStart : Nat := 158461
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩] } }
def leftRaw : List Term := Proof.Events621.exact159136RawTerms
def rightRaw : List Term := Proof.Events621.exact158977RawTerms
def group : MergeGroup := .operator 159136 158977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 159136) (leftOrdinal := 20)
    (rightResult := 158977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22029⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71142⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22029⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge159203

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
