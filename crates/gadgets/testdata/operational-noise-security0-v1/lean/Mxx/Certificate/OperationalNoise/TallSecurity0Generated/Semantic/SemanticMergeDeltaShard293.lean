import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge49249
def owner : Owner := ⟨.program ⟨214⟩, ⟨15668⟩⟩
def mergeEvent : Nat := 49249
def frameStart : Nat := 49183
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49245RawTerms
def rightRaw : List Term := Proof.Events192.exact49243RawTerms
def group : MergeGroup := .operator 49245 49243
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49245) (leftOrdinal := 0)
    (rightResult := 49243) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15591⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49249

namespace LeftMerge49261
def owner : Owner := ⟨.program ⟨214⟩, ⟨27235⟩⟩
def mergeEvent : Nat := 49261
def frameStart : Nat := 49183
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49257RawTerms
def rightRaw : List Term := Proof.Events192.exact49234RawTerms
def group : MergeGroup := .operator 49257 49234
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49257) (leftOrdinal := 0)
    (rightResult := 49234) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27234⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49261

namespace LeftMerge49262
def owner : Owner := ⟨.program ⟨214⟩, ⟨27235⟩⟩
def mergeEvent : Nat := 49262
def frameStart : Nat := 49183
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49257RawTerms
def rightRaw : List Term := Proof.Events192.exact49234RawTerms
def group : MergeGroup := .operator 49257 49234
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49257) (leftOrdinal := 1)
    (rightResult := 49234) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27234⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15591⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49262

namespace LeftMerge49264
def owner : Owner := ⟨.program ⟨214⟩, ⟨27235⟩⟩
def mergeEvent : Nat := 49264
def frameStart : Nat := 49183
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23978⟩⟩] } }
def rhsRaw : List Term := Proof.Events192.exact49231RawTerms
def group : MergeGroup := .relation 49263
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49263) (rhsResult := 49231)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15591⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27234⟩⟩) ⟨23978⟩ 49231) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23978⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15591⟩⟩], [⟨.program ⟨214⟩, ⟨23978⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49264

namespace LeftMerge49272
def owner : Owner := ⟨.program ⟨214⟩, ⟨17836⟩⟩
def mergeEvent : Nat := 49272
def frameStart : Nat := 49183
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49245RawTerms
def rightRaw : List Term := Proof.Events192.exact49268RawTerms
def group : MergeGroup := .operator 49245 49268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49245) (leftOrdinal := 0)
    (rightResult := 49268) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17830⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49272

namespace LeftMerge49289
def owner : Owner := ⟨.program ⟨214⟩, ⟨20907⟩⟩
def mergeEvent : Nat := 49289
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩] } }
def rhsRaw : List Term := Proof.Events192.exact49286RawTerms
def group : MergeGroup := .relation 49288
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49288) (rhsResult := 49286)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20904⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49287 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20904⟩⟩]⟩) (none) 49286) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49289

namespace LeftMerge49290
def owner : Owner := ⟨.program ⟨214⟩, ⟨20907⟩⟩
def mergeEvent : Nat := 49290
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩] } }
def rhsRaw : List Term := Proof.Events192.exact49286RawTerms
def group : MergeGroup := .relation 49288
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49288) (rhsResult := 49286)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20904⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49287 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20904⟩⟩]⟩) (none) 49286) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49290

namespace LeftMerge49291
def owner : Owner := ⟨.program ⟨214⟩, ⟨20907⟩⟩
def mergeEvent : Nat := 49291
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23978⟩⟩] } }
def rhsRaw : List Term := Proof.Events192.exact49286RawTerms
def group : MergeGroup := .relation 49288
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49288) (rhsResult := 49286)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20904⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49287 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20904⟩⟩]⟩) (none) 49286) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23978⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], [⟨.program ⟨214⟩, ⟨23978⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49291

namespace LeftMerge49292
def owner : Owner := ⟨.program ⟨214⟩, ⟨20907⟩⟩
def mergeEvent : Nat := 49292
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events192.exact49286RawTerms
def group : MergeGroup := .relation 49288
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49288) (rhsResult := 49286)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20904⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 49287 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20904⟩⟩]⟩) (none) 49286) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49292

namespace LeftMerge49297
def owner : Owner := ⟨.program ⟨214⟩, ⟨27237⟩⟩
def mergeEvent : Nat := 49297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49293RawTerms
def rightRaw : List Term := Proof.Events191.exact49115RawTerms
def group : MergeGroup := .operator 49293 49115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49293) (leftOrdinal := 0)
    (rightResult := 49115) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27234⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49297

namespace LeftMerge49298
def owner : Owner := ⟨.program ⟨214⟩, ⟨27237⟩⟩
def mergeEvent : Nat := 49298
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23978⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49293RawTerms
def rightRaw : List Term := Proof.Events191.exact49115RawTerms
def group : MergeGroup := .operator 49293 49115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49293) (leftOrdinal := 2)
    (rightResult := 49115) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23978⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23978⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], [⟨.program ⟨214⟩, ⟨23978⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49298

namespace LeftMerge49306
def owner : Owner := ⟨.program ⟨214⟩, ⟨27238⟩⟩
def mergeEvent : Nat := 49306
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49300RawTerms
def rightRaw : List Term := Proof.Events022.exact5779RawTerms
def group : MergeGroup := .operator 49300 5779
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49300) (leftOrdinal := 0)
    (rightResult := 5779) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6716⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6649⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6716⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49306

namespace LeftMerge49307
def owner : Owner := ⟨.program ⟨214⟩, ⟨27238⟩⟩
def mergeEvent : Nat := 49307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩] } }
def leftRaw : List Term := Proof.Events192.exact49300RawTerms
def rightRaw : List Term := Proof.Events022.exact5779RawTerms
def group : MergeGroup := .operator 49300 5779
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49300) (leftOrdinal := 1)
    (rightResult := 5779) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6649⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49307

namespace LeftMerge49309
def owner : Owner := ⟨.program ⟨214⟩, ⟨27238⟩⟩
def mergeEvent : Nat := 49309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events022.exact5772RawTerms
def group : MergeGroup := .relation 49308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49308) (rhsResult := 5772)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6649⟩⟩) ⟨6596⟩ 5772) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6407⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6407⟩⟩, ⟨.program ⟨214⟩, ⟨17830⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49309

namespace LeftMerge49323
def owner : Owner := ⟨.program ⟨214⟩, ⟨27019⟩⟩
def mergeEvent : Nat := 49323
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27017⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43071RawTerms
def rightRaw : List Term := Proof.Events192.exact49317RawTerms
def group : MergeGroup := .operator 43071 49317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43071) (leftOrdinal := 0)
    (rightResult := 49317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27017⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27017⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49323

namespace LeftMerge49324
def owner : Owner := ⟨.program ⟨214⟩, ⟨27019⟩⟩
def mergeEvent : Nat := 49324
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27017⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43071RawTerms
def rightRaw : List Term := Proof.Events192.exact49317RawTerms
def group : MergeGroup := .operator 43071 49317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43071) (leftOrdinal := 1)
    (rightResult := 49317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27017⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27017⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49324

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
