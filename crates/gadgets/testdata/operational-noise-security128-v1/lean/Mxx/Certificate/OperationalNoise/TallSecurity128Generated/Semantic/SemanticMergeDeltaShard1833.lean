import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge296186
def owner : Owner := ⟨.program ⟨257⟩, ⟨44192⟩⟩
def mergeEvent : Nat := 296186
def frameStart : Nat := 296106
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩] } }
def leftRaw : List Term := Proof.Events1156.exact296182RawTerms
def rightRaw : List Term := Proof.Events1156.exact296139RawTerms
def group : MergeGroup := .operator 296182 296139
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296182) (leftOrdinal := 0)
    (rightResult := 296139) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44189⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296186

namespace LeftMerge296187
def owner : Owner := ⟨.program ⟨257⟩, ⟨44192⟩⟩
def mergeEvent : Nat := 296187
def frameStart : Nat := 296106
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩] } }
def leftRaw : List Term := Proof.Events1156.exact296182RawTerms
def rightRaw : List Term := Proof.Events1156.exact296139RawTerms
def group : MergeGroup := .operator 296182 296139
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296182) (leftOrdinal := 1)
    (rightResult := 296139) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44189⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296187

namespace LeftMerge296189
def owner : Owner := ⟨.program ⟨257⟩, ⟨44192⟩⟩
def mergeEvent : Nat := 296189
def frameStart : Nat := 296106
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43729⟩⟩] } }
def rhsRaw : List Term := Proof.Events1156.exact296136RawTerms
def group : MergeGroup := .relation 296188
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296188) (rhsResult := 296136)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44189⟩⟩) ⟨43729⟩ 296136) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43729⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], [⟨.program ⟨257⟩, ⟨43729⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296189

namespace LeftMerge296197
def owner : Owner := ⟨.program ⟨257⟩, ⟨42710⟩⟩
def mergeEvent : Nat := 296197
def frameStart : Nat := 296106
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1156.exact296150RawTerms
def rightRaw : List Term := Proof.Events1157.exact296193RawTerms
def group : MergeGroup := .operator 296150 296193
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296150) (leftOrdinal := 0)
    (rightResult := 296193) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42708⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296197

namespace LeftMerge296214
def owner : Owner := ⟨.program ⟨257⟩, ⟨43132⟩⟩
def mergeEvent : Nat := 296214
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events1157.exact296211RawTerms
def group : MergeGroup := .relation 296213
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296213) (rhsResult := 296211)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43129⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 296212 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43129⟩⟩]⟩) (none) 296211) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296214

namespace LeftMerge296215
def owner : Owner := ⟨.program ⟨257⟩, ⟨43132⟩⟩
def mergeEvent : Nat := 296215
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩] } }
def rhsRaw : List Term := Proof.Events1157.exact296211RawTerms
def group : MergeGroup := .relation 296213
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296213) (rhsResult := 296211)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43129⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 296212 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43129⟩⟩]⟩) (none) 296211) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296215

namespace LeftMerge296216
def owner : Owner := ⟨.program ⟨257⟩, ⟨43132⟩⟩
def mergeEvent : Nat := 296216
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43729⟩⟩] } }
def rhsRaw : List Term := Proof.Events1157.exact296211RawTerms
def group : MergeGroup := .relation 296213
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296213) (rhsResult := 296211)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43129⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 296212 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43129⟩⟩]⟩) (none) 296211) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43729⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], [⟨.program ⟨257⟩, ⟨43729⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296216

namespace LeftMerge296217
def owner : Owner := ⟨.program ⟨257⟩, ⟨43132⟩⟩
def mergeEvent : Nat := 296217
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1157.exact296211RawTerms
def group : MergeGroup := .relation 296213
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296213) (rhsResult := 296211)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43129⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 296212 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43129⟩⟩]⟩) (none) 296211) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296217

namespace LeftMerge296222
def owner : Owner := ⟨.program ⟨257⟩, ⟨44191⟩⟩
def mergeEvent : Nat := 296222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43729⟩⟩] } }
def leftRaw : List Term := Proof.Events1157.exact296218RawTerms
def rightRaw : List Term := Proof.Events1156.exact296056RawTerms
def group : MergeGroup := .operator 296218 296056
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296218) (leftOrdinal := 2)
    (rightResult := 296056) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43729⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43729⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨14331⟩⟩, ⟨.program ⟨257⟩, ⟨42234⟩⟩], [⟨.program ⟨257⟩, ⟨43729⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296222

namespace LeftMerge296223
def owner : Owner := ⟨.program ⟨257⟩, ⟨44191⟩⟩
def mergeEvent : Nat := 296223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩] } }
def leftRaw : List Term := Proof.Events1157.exact296218RawTerms
def rightRaw : List Term := Proof.Events1156.exact296056RawTerms
def group : MergeGroup := .operator 296218 296056
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296218) (leftOrdinal := 1)
    (rightResult := 296056) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296223

namespace LeftMerge296231
def owner : Owner := ⟨.program ⟨257⟩, ⟨44421⟩⟩
def mergeEvent : Nat := 296231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44419⟩⟩] } }
def leftRaw : List Term := Proof.Events1157.exact296225RawTerms
def rightRaw : List Term := Proof.Events1156.exact295972RawTerms
def group : MergeGroup := .operator 296225 295972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296225) (leftOrdinal := 0)
    (rightResult := 295972) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44419⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44419⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296231

namespace LeftMerge296232
def owner : Owner := ⟨.program ⟨257⟩, ⟨44421⟩⟩
def mergeEvent : Nat := 296232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44419⟩⟩] } }
def leftRaw : List Term := Proof.Events1157.exact296225RawTerms
def rightRaw : List Term := Proof.Events1156.exact295972RawTerms
def group : MergeGroup := .operator 296225 295972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296225) (leftOrdinal := 1)
    (rightResult := 295972) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44419⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44419⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296232

namespace LeftMerge296234
def owner : Owner := ⟨.program ⟨257⟩, ⟨44421⟩⟩
def mergeEvent : Nat := 296234
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43851⟩⟩] } }
def rhsRaw : List Term := Proof.Events1156.exact295969RawTerms
def group : MergeGroup := .relation 296233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 296233) (rhsResult := 295969)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44419⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44419⟩⟩) ⟨43851⟩ 295969) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43851⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42708⟩⟩], [⟨.program ⟨257⟩, ⟨43851⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge296234

namespace LeftMerge296248
def owner : Owner := ⟨.program ⟨257⟩, ⟨43339⟩⟩
def mergeEvent : Nat := 296248
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43336⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1157.exact296242RawTerms
def group : MergeGroup := .operator 295195 296242
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 296242) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43336⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43336⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296248

namespace LeftMerge296345
def owner : Owner := ⟨.program ⟨257⟩, ⟨44108⟩⟩
def mergeEvent : Nat := 296345
def frameStart : Nat := 296291
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1157.exact296341RawTerms
def rightRaw : List Term := Proof.Events1157.exact296339RawTerms
def group : MergeGroup := .operator 296341 296339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296341) (leftOrdinal := 0)
    (rightResult := 296339) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42708⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296345

namespace LeftMerge296357
def owner : Owner := ⟨.program ⟨257⟩, ⟨44420⟩⟩
def mergeEvent : Nat := 296357
def frameStart : Nat := 296291
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44419⟩⟩] } }
def leftRaw : List Term := Proof.Events1157.exact296353RawTerms
def rightRaw : List Term := Proof.Events1157.exact296330RawTerms
def group : MergeGroup := .operator 296353 296330
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 296353) (leftOrdinal := 0)
    (rightResult := 296330) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44419⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44419⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge296357

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
