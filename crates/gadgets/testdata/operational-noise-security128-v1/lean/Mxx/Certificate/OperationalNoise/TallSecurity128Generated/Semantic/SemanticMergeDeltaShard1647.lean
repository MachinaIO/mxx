import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge266296
def owner : Owner := ⟨.program ⟨257⟩, ⟨48509⟩⟩
def mergeEvent : Nat := 266296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49568⟩⟩] } }
def rhsRaw : List Term := Proof.Events1040.exact266292RawTerms
def group : MergeGroup := .relation 266294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 266294) (rhsResult := 266292)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48506⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 266293 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48506⟩⟩]⟩) (none) 266292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49568⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49568⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge266296

namespace LeftMerge266297
def owner : Owner := ⟨.program ⟨257⟩, ⟨48509⟩⟩
def mergeEvent : Nat := 266297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49099⟩⟩] } }
def rhsRaw : List Term := Proof.Events1040.exact266292RawTerms
def group : MergeGroup := .relation 266294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 266294) (rhsResult := 266292)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48506⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 266293 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48506⟩⟩]⟩) (none) 266292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49099⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], [⟨.program ⟨257⟩, ⟨49099⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge266297

namespace LeftMerge266298
def owner : Owner := ⟨.program ⟨257⟩, ⟨48509⟩⟩
def mergeEvent : Nat := 266298
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1040.exact266292RawTerms
def group : MergeGroup := .relation 266294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 266294) (rhsResult := 266292)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48506⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 266293 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48506⟩⟩]⟩) (none) 266292) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge266298

namespace LeftMerge266303
def owner : Owner := ⟨.program ⟨257⟩, ⟨49570⟩⟩
def mergeEvent : Nat := 266303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49099⟩⟩] } }
def leftRaw : List Term := Proof.Events1040.exact266299RawTerms
def rightRaw : List Term := Proof.Events1039.exact266102RawTerms
def group : MergeGroup := .operator 266299 266102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266299) (leftOrdinal := 2)
    (rightResult := 266102) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49099⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49099⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14956⟩⟩, ⟨.program ⟨257⟩, ⟨47634⟩⟩], [⟨.program ⟨257⟩, ⟨49099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge266303

namespace LeftMerge266304
def owner : Owner := ⟨.program ⟨257⟩, ⟨49570⟩⟩
def mergeEvent : Nat := 266304
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49568⟩⟩] } }
def leftRaw : List Term := Proof.Events1040.exact266299RawTerms
def rightRaw : List Term := Proof.Events1039.exact266102RawTerms
def group : MergeGroup := .operator 266299 266102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266299) (leftOrdinal := 1)
    (rightResult := 266102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49568⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge266304

namespace LeftMerge266312
def owner : Owner := ⟨.program ⟨257⟩, ⟨49824⟩⟩
def mergeEvent : Nat := 266312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩] } }
def leftRaw : List Term := Proof.Events1040.exact266306RawTerms
def rightRaw : List Term := Proof.Events1039.exact266013RawTerms
def group : MergeGroup := .operator 266306 266013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266306) (leftOrdinal := 0)
    (rightResult := 266013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49822⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge266312

namespace LeftMerge266313
def owner : Owner := ⟨.program ⟨257⟩, ⟨49824⟩⟩
def mergeEvent : Nat := 266313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩] } }
def leftRaw : List Term := Proof.Events1040.exact266306RawTerms
def rightRaw : List Term := Proof.Events1039.exact266013RawTerms
def group : MergeGroup := .operator 266306 266013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266306) (leftOrdinal := 1)
    (rightResult := 266013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49822⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge266313

namespace LeftMerge266315
def owner : Owner := ⟨.program ⟨257⟩, ⟨49824⟩⟩
def mergeEvent : Nat := 266315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49226⟩⟩] } }
def rhsRaw : List Term := Proof.Events1039.exact266010RawTerms
def group : MergeGroup := .relation 266314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 266314) (rhsResult := 266010)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49822⟩⟩) ⟨49226⟩ 266010) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49226⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49226⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge266315

namespace LeftMerge266329
def owner : Owner := ⟨.program ⟨257⟩, ⟨48733⟩⟩
def mergeEvent : Nat := 266329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48730⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1040.exact266323RawTerms
def group : MergeGroup := .operator 266120 266323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 266323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48730⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48730⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge266329

namespace LeftMerge266450
def owner : Owner := ⟨.program ⟨257⟩, ⟨49476⟩⟩
def mergeEvent : Nat := 266450
def frameStart : Nat := 266384
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1040.exact266446RawTerms
def rightRaw : List Term := Proof.Events1040.exact266444RawTerms
def group : MergeGroup := .operator 266446 266444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266446) (leftOrdinal := 0)
    (rightResult := 266444) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge266450

namespace LeftMerge266462
def owner : Owner := ⟨.program ⟨257⟩, ⟨49823⟩⟩
def mergeEvent : Nat := 266462
def frameStart : Nat := 266384
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩] } }
def leftRaw : List Term := Proof.Events1040.exact266458RawTerms
def rightRaw : List Term := Proof.Events1040.exact266435RawTerms
def group : MergeGroup := .operator 266458 266435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266458) (leftOrdinal := 0)
    (rightResult := 266435) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49822⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge266462

namespace LeftMerge266463
def owner : Owner := ⟨.program ⟨257⟩, ⟨49823⟩⟩
def mergeEvent : Nat := 266463
def frameStart : Nat := 266384
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩] } }
def leftRaw : List Term := Proof.Events1040.exact266458RawTerms
def rightRaw : List Term := Proof.Events1040.exact266435RawTerms
def group : MergeGroup := .operator 266458 266435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266458) (leftOrdinal := 1)
    (rightResult := 266435) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49822⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge266463

namespace LeftMerge266465
def owner : Owner := ⟨.program ⟨257⟩, ⟨49823⟩⟩
def mergeEvent : Nat := 266465
def frameStart : Nat := 266384
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48082⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49226⟩⟩] } }
def rhsRaw : List Term := Proof.Events1040.exact266432RawTerms
def group : MergeGroup := .relation 266464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 266464) (rhsResult := 266432)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49822⟩⟩) ⟨49226⟩ 266432) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49226⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48082⟩⟩], [⟨.program ⟨257⟩, ⟨49226⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge266465

namespace LeftMerge266473
def owner : Owner := ⟨.program ⟨257⟩, ⟨48257⟩⟩
def mergeEvent : Nat := 266473
def frameStart : Nat := 266384
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1040.exact266446RawTerms
def rightRaw : List Term := Proof.Events1040.exact266469RawTerms
def group : MergeGroup := .operator 266446 266469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266446) (leftOrdinal := 0)
    (rightResult := 266469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48256⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge266473

namespace LeftMerge266490
def owner : Owner := ⟨.program ⟨257⟩, ⟨48733⟩⟩
def mergeEvent : Nat := 266490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }
def rhsRaw : List Term := Proof.Events1040.exact266487RawTerms
def group : MergeGroup := .relation 266489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 266489) (rhsResult := 266487)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48730⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 266488 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48730⟩⟩]⟩) (none) 266487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge266490

namespace LeftMerge266491
def owner : Owner := ⟨.program ⟨257⟩, ⟨48733⟩⟩
def mergeEvent : Nat := 266491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩] } }
def rhsRaw : List Term := Proof.Events1040.exact266487RawTerms
def group : MergeGroup := .relation 266489
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 266489) (rhsResult := 266487)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48730⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 266488 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48730⟩⟩]⟩) (none) 266487) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49822⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge266491

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
