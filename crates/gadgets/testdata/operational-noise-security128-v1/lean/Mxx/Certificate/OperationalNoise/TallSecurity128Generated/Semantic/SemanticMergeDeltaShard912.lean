import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge149205
def owner : Owner := ⟨.program ⟨257⟩, ⟨47763⟩⟩
def mergeEvent : Nat := 149205
def frameStart : Nat := 149175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events582.exact149201RawTerms
def rightRaw : List Term := Proof.Events582.exact149198RawTerms
def group : MergeGroup := .operator 149201 149198
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149201) (leftOrdinal := 0)
    (rightResult := 149198) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15036⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149205

namespace LeftMerge149235
def owner : Owner := ⟨.program ⟨257⟩, ⟨49416⟩⟩
def mergeEvent : Nat := 149235
def frameStart : Nat := 149175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149231RawTerms
def rightRaw : List Term := Proof.Events582.exact149229RawTerms
def group : MergeGroup := .operator 149231 149229
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149231) (leftOrdinal := 0)
    (rightResult := 149229) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149235

namespace LeftMerge149258
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 149258
def frameStart : Nat := 149175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events583.exact149254RawTerms
def rightRaw : List Term := Proof.Events583.exact149251RawTerms
def group : MergeGroup := .operator 149254 149251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149254) (leftOrdinal := 0)
    (rightResult := 149251) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149258

namespace LeftMerge149267
def owner : Owner := ⟨.program ⟨257⟩, ⟨49629⟩⟩
def mergeEvent : Nat := 149267
def frameStart : Nat := 149175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩] } }
def leftRaw : List Term := Proof.Events583.exact149263RawTerms
def rightRaw : List Term := Proof.Events582.exact149220RawTerms
def group : MergeGroup := .operator 149263 149220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149263) (leftOrdinal := 0)
    (rightResult := 149220) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49626⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149267

namespace LeftMerge149268
def owner : Owner := ⟨.program ⟨257⟩, ⟨49629⟩⟩
def mergeEvent : Nat := 149268
def frameStart : Nat := 149175
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩] } }
def leftRaw : List Term := Proof.Events583.exact149263RawTerms
def rightRaw : List Term := Proof.Events582.exact149220RawTerms
def group : MergeGroup := .operator 149263 149220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149263) (leftOrdinal := 1)
    (rightResult := 149220) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49626⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149268

namespace LeftMerge149270
def owner : Owner := ⟨.program ⟨257⟩, ⟨49629⟩⟩
def mergeEvent : Nat := 149270
def frameStart : Nat := 149175
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49131⟩⟩] } }
def rhsRaw : List Term := Proof.Events582.exact149217RawTerms
def group : MergeGroup := .relation 149269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 149269) (rhsResult := 149217)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49626⟩⟩) ⟨49131⟩ 149217) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49131⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], [⟨.program ⟨257⟩, ⟨49131⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149270

namespace LeftMerge149278
def owner : Owner := ⟨.program ⟨257⟩, ⟨48126⟩⟩
def mergeEvent : Nat := 149278
def frameStart : Nat := 149175
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149231RawTerms
def rightRaw : List Term := Proof.Events583.exact149274RawTerms
def group : MergeGroup := .operator 149231 149274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149231) (leftOrdinal := 0)
    (rightResult := 149274) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48124⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149278

namespace LeftMerge149295
def owner : Owner := ⟨.program ⟨257⟩, ⟨48562⟩⟩
def mergeEvent : Nat := 149295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events583.exact149292RawTerms
def group : MergeGroup := .relation 149294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 149294) (rhsResult := 149292)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 149293 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48559⟩⟩]⟩) (none) 149292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149295

namespace LeftMerge149296
def owner : Owner := ⟨.program ⟨257⟩, ⟨48562⟩⟩
def mergeEvent : Nat := 149296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩] } }
def rhsRaw : List Term := Proof.Events583.exact149292RawTerms
def group : MergeGroup := .relation 149294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 149294) (rhsResult := 149292)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 149293 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48559⟩⟩]⟩) (none) 149292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149296

namespace LeftMerge149297
def owner : Owner := ⟨.program ⟨257⟩, ⟨48562⟩⟩
def mergeEvent : Nat := 149297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49131⟩⟩] } }
def rhsRaw : List Term := Proof.Events583.exact149292RawTerms
def group : MergeGroup := .relation 149294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 149294) (rhsResult := 149292)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 149293 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48559⟩⟩]⟩) (none) 149292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49131⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], [⟨.program ⟨257⟩, ⟨49131⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149297

namespace LeftMerge149298
def owner : Owner := ⟨.program ⟨257⟩, ⟨48562⟩⟩
def mergeEvent : Nat := 149298
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events583.exact149292RawTerms
def group : MergeGroup := .relation 149294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 149294) (rhsResult := 149292)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 149293 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48559⟩⟩]⟩) (none) 149292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149298

namespace LeftMerge149303
def owner : Owner := ⟨.program ⟨257⟩, ⟨49628⟩⟩
def mergeEvent : Nat := 149303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49131⟩⟩] } }
def leftRaw : List Term := Proof.Events583.exact149299RawTerms
def rightRaw : List Term := Proof.Events582.exact149102RawTerms
def group : MergeGroup := .operator 149299 149102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149299) (leftOrdinal := 2)
    (rightResult := 149102) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49131⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨15036⟩⟩, ⟨.program ⟨257⟩, ⟨47762⟩⟩], [⟨.program ⟨257⟩, ⟨49131⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149303

namespace LeftMerge149304
def owner : Owner := ⟨.program ⟨257⟩, ⟨49628⟩⟩
def mergeEvent : Nat := 149304
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩] } }
def leftRaw : List Term := Proof.Events583.exact149299RawTerms
def rightRaw : List Term := Proof.Events582.exact149102RawTerms
def group : MergeGroup := .operator 149299 149102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149299) (leftOrdinal := 1)
    (rightResult := 149102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49626⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149304

namespace LeftMerge149312
def owner : Owner := ⟨.program ⟨257⟩, ⟨49956⟩⟩
def mergeEvent : Nat := 149312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49954⟩⟩] } }
def leftRaw : List Term := Proof.Events583.exact149306RawTerms
def rightRaw : List Term := Proof.Events582.exact149013RawTerms
def group : MergeGroup := .operator 149306 149013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149306) (leftOrdinal := 0)
    (rightResult := 149013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49954⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49954⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge149312

namespace LeftMerge149313
def owner : Owner := ⟨.program ⟨257⟩, ⟨49956⟩⟩
def mergeEvent : Nat := 149313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49954⟩⟩] } }
def leftRaw : List Term := Proof.Events583.exact149306RawTerms
def rightRaw : List Term := Proof.Events582.exact149013RawTerms
def group : MergeGroup := .operator 149306 149013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149306) (leftOrdinal := 1)
    (rightResult := 149013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49954⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49954⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149313

namespace LeftMerge149315
def owner : Owner := ⟨.program ⟨257⟩, ⟨49956⟩⟩
def mergeEvent : Nat := 149315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49274⟩⟩] } }
def rhsRaw : List Term := Proof.Events582.exact149010RawTerms
def group : MergeGroup := .relation 149314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 149314) (rhsResult := 149010)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49954⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49954⟩⟩) ⟨49274⟩ 149010) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49274⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨48124⟩⟩], [⟨.program ⟨257⟩, ⟨49274⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge149315

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
