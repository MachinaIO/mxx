import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge267232
def owner : Owner := ⟨.program ⟨257⟩, ⟨44211⟩⟩
def mergeEvent : Nat := 267232
def frameStart : Nat := 267139
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267227RawTerms
def rightRaw : List Term := Proof.Events1043.exact267184RawTerms
def group : MergeGroup := .operator 267227 267184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267227) (leftOrdinal := 1)
    (rightResult := 267184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44208⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267232

namespace LeftMerge267234
def owner : Owner := ⟨.program ⟨257⟩, ⟨44211⟩⟩
def mergeEvent : Nat := 267234
def frameStart : Nat := 267139
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43739⟩⟩] } }
def rhsRaw : List Term := Proof.Events1043.exact267181RawTerms
def group : MergeGroup := .relation 267233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267233) (rhsResult := 267181)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44208⟩⟩) ⟨43739⟩ 267181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43739⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨43739⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267234

namespace LeftMerge267242
def owner : Owner := ⟨.program ⟨257⟩, ⟨42724⟩⟩
def mergeEvent : Nat := 267242
def frameStart : Nat := 267139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267195RawTerms
def rightRaw : List Term := Proof.Events1043.exact267238RawTerms
def group : MergeGroup := .operator 267195 267238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267195) (leftOrdinal := 0)
    (rightResult := 267238) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267242

namespace LeftMerge267259
def owner : Owner := ⟨.program ⟨257⟩, ⟨43149⟩⟩
def mergeEvent : Nat := 267259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events1043.exact267256RawTerms
def group : MergeGroup := .relation 267258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267258) (rhsResult := 267256)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 267257 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩]⟩) (none) 267256) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267259

namespace LeftMerge267260
def owner : Owner := ⟨.program ⟨257⟩, ⟨43149⟩⟩
def mergeEvent : Nat := 267260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩] } }
def rhsRaw : List Term := Proof.Events1043.exact267256RawTerms
def group : MergeGroup := .relation 267258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267258) (rhsResult := 267256)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 267257 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩]⟩) (none) 267256) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267260

namespace LeftMerge267261
def owner : Owner := ⟨.program ⟨257⟩, ⟨43149⟩⟩
def mergeEvent : Nat := 267261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43739⟩⟩] } }
def rhsRaw : List Term := Proof.Events1043.exact267256RawTerms
def group : MergeGroup := .relation 267258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267258) (rhsResult := 267256)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 267257 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩]⟩) (none) 267256) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43739⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨43739⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267261

namespace LeftMerge267262
def owner : Owner := ⟨.program ⟨257⟩, ⟨43149⟩⟩
def mergeEvent : Nat := 267262
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1043.exact267256RawTerms
def group : MergeGroup := .relation 267258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267258) (rhsResult := 267256)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 267257 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩]⟩) (none) 267256) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267262

namespace LeftMerge267267
def owner : Owner := ⟨.program ⟨257⟩, ⟨44210⟩⟩
def mergeEvent : Nat := 267267
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43739⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267263RawTerms
def rightRaw : List Term := Proof.Events1043.exact267077RawTerms
def group : MergeGroup := .operator 267263 267077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267263) (leftOrdinal := 2)
    (rightResult := 267077) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43739⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43739⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨43739⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267267

namespace LeftMerge267268
def owner : Owner := ⟨.program ⟨257⟩, ⟨44210⟩⟩
def mergeEvent : Nat := 267268
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267263RawTerms
def rightRaw : List Term := Proof.Events1043.exact267077RawTerms
def group : MergeGroup := .operator 267263 267077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267263) (leftOrdinal := 1)
    (rightResult := 267077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267268

namespace LeftMerge267276
def owner : Owner := ⟨.program ⟨257⟩, ⟨44464⟩⟩
def mergeEvent : Nat := 267276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44462⟩⟩] } }
def leftRaw : List Term := Proof.Events1044.exact267270RawTerms
def rightRaw : List Term := Proof.Events1042.exact266993RawTerms
def group : MergeGroup := .operator 267270 266993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267270) (leftOrdinal := 0)
    (rightResult := 266993) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44462⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44462⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267276

namespace LeftMerge267277
def owner : Owner := ⟨.program ⟨257⟩, ⟨44464⟩⟩
def mergeEvent : Nat := 267277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44462⟩⟩] } }
def leftRaw : List Term := Proof.Events1044.exact267270RawTerms
def rightRaw : List Term := Proof.Events1042.exact266993RawTerms
def group : MergeGroup := .operator 267270 266993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267270) (leftOrdinal := 1)
    (rightResult := 266993) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44462⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44462⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267277

namespace LeftMerge267279
def owner : Owner := ⟨.program ⟨257⟩, ⟨44464⟩⟩
def mergeEvent : Nat := 267279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43866⟩⟩] } }
def rhsRaw : List Term := Proof.Events1042.exact266990RawTerms
def group : MergeGroup := .relation 267278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267278) (rhsResult := 266990)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44462⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44462⟩⟩) ⟨43866⟩ 266990) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42722⟩⟩], [⟨.program ⟨257⟩, ⟨43866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267279

namespace LeftMerge267293
def owner : Owner := ⟨.program ⟨257⟩, ⟨43373⟩⟩
def mergeEvent : Nat := 267293
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43370⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1044.exact267287RawTerms
def group : MergeGroup := .operator 266120 267287
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 267287) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43370⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43370⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267293

namespace LeftMerge267414
def owner : Owner := ⟨.program ⟨257⟩, ⟨44116⟩⟩
def mergeEvent : Nat := 267414
def frameStart : Nat := 267348
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1044.exact267410RawTerms
def rightRaw : List Term := Proof.Events1044.exact267408RawTerms
def group : MergeGroup := .operator 267410 267408
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267410) (leftOrdinal := 0)
    (rightResult := 267408) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267414

namespace LeftMerge267426
def owner : Owner := ⟨.program ⟨257⟩, ⟨44463⟩⟩
def mergeEvent : Nat := 267426
def frameStart : Nat := 267348
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44462⟩⟩] } }
def leftRaw : List Term := Proof.Events1044.exact267422RawTerms
def rightRaw : List Term := Proof.Events1044.exact267399RawTerms
def group : MergeGroup := .operator 267422 267399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267422) (leftOrdinal := 0)
    (rightResult := 267399) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44462⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44462⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267426

namespace LeftMerge267427
def owner : Owner := ⟨.program ⟨257⟩, ⟨44463⟩⟩
def mergeEvent : Nat := 267427
def frameStart : Nat := 267348
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44462⟩⟩] } }
def leftRaw : List Term := Proof.Events1044.exact267422RawTerms
def rightRaw : List Term := Proof.Events1044.exact267399RawTerms
def group : MergeGroup := .operator 267422 267399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267422) (leftOrdinal := 1)
    (rightResult := 267399) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44462⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44462⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267427

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
