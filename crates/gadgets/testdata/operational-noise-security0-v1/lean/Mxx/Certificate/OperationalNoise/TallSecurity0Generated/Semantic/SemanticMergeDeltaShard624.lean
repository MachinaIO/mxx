import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge101157
def owner : Owner := ⟨.program ⟨214⟩, ⟨20672⟩⟩
def mergeEvent : Nat := 101157
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events395.exact101151RawTerms
def group : MergeGroup := .operator 94462 101151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 101151) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20669⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101157

namespace LeftMerge101254
def owner : Owner := ⟨.program ⟨214⟩, ⟨15148⟩⟩
def mergeEvent : Nat := 101254
def frameStart : Nat := 101200
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101250RawTerms
def rightRaw : List Term := Proof.Events395.exact101248RawTerms
def group : MergeGroup := .operator 101250 101248
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101250) (leftOrdinal := 0)
    (rightResult := 101248) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101254

namespace LeftMerge101266
def owner : Owner := ⟨.program ⟨214⟩, ⟨26747⟩⟩
def mergeEvent : Nat := 101266
def frameStart : Nat := 101200
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101262RawTerms
def rightRaw : List Term := Proof.Events395.exact101239RawTerms
def group : MergeGroup := .operator 101262 101239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101262) (leftOrdinal := 0)
    (rightResult := 101239) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26746⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101266

namespace LeftMerge101267
def owner : Owner := ⟨.program ⟨214⟩, ⟨26747⟩⟩
def mergeEvent : Nat := 101267
def frameStart : Nat := 101200
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101262RawTerms
def rightRaw : List Term := Proof.Events395.exact101239RawTerms
def group : MergeGroup := .operator 101262 101239
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101262) (leftOrdinal := 1)
    (rightResult := 101239) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101267

namespace LeftMerge101269
def owner : Owner := ⟨.program ⟨214⟩, ⟨26747⟩⟩
def mergeEvent : Nat := 101269
def frameStart : Nat := 101200
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23838⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101236RawTerms
def group : MergeGroup := .relation 101268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101268) (rhsResult := 101236)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26746⟩⟩) ⟨23838⟩ 101236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23838⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23838⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101269

namespace LeftMerge101277
def owner : Owner := ⟨.program ⟨214⟩, ⟨15358⟩⟩
def mergeEvent : Nat := 101277
def frameStart : Nat := 101200
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101250RawTerms
def rightRaw : List Term := Proof.Events395.exact101273RawTerms
def group : MergeGroup := .operator 101250 101273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101250) (leftOrdinal := 0)
    (rightResult := 101273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101277

namespace LeftMerge101294
def owner : Owner := ⟨.program ⟨214⟩, ⟨20672⟩⟩
def mergeEvent : Nat := 101294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101291RawTerms
def group : MergeGroup := .relation 101293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101293) (rhsResult := 101291)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101292 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩) (none) 101291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101294

namespace LeftMerge101295
def owner : Owner := ⟨.program ⟨214⟩, ⟨20672⟩⟩
def mergeEvent : Nat := 101295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101291RawTerms
def group : MergeGroup := .relation 101293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101293) (rhsResult := 101291)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101292 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩) (none) 101291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101295

namespace LeftMerge101296
def owner : Owner := ⟨.program ⟨214⟩, ⟨20672⟩⟩
def mergeEvent : Nat := 101296
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23838⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101291RawTerms
def group : MergeGroup := .relation 101293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101293) (rhsResult := 101291)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101292 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩) (none) 101291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23838⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23838⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101296

namespace LeftMerge101297
def owner : Owner := ⟨.program ⟨214⟩, ⟨20672⟩⟩
def mergeEvent : Nat := 101297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events395.exact101291RawTerms
def group : MergeGroup := .relation 101293
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 101293) (rhsResult := 101291)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 101292 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩) (none) 101291) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15356⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15356⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101297

namespace LeftMerge101302
def owner : Owner := ⟨.program ⟨214⟩, ⟨26749⟩⟩
def mergeEvent : Nat := 101302
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101298RawTerms
def rightRaw : List Term := Proof.Events395.exact101144RawTerms
def group : MergeGroup := .operator 101298 101144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101298) (leftOrdinal := 0)
    (rightResult := 101144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6692⟩⟩, ⟨.program ⟨214⟩, ⟨26746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101302

namespace LeftMerge101303
def owner : Owner := ⟨.program ⟨214⟩, ⟨26749⟩⟩
def mergeEvent : Nat := 101303
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23838⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101298RawTerms
def rightRaw : List Term := Proof.Events395.exact101144RawTerms
def group : MergeGroup := .operator 101298 101144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101298) (leftOrdinal := 2)
    (rightResult := 101144) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23838⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23838⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15104⟩⟩], [⟨.program ⟨214⟩, ⟨23838⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101303

namespace LeftMerge101329
def owner : Owner := ⟨.program ⟨214⟩, ⟨10655⟩⟩
def mergeEvent : Nat := 101329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events019.exact4934RawTerms
def rightRaw : List Term := Proof.Events000.exact32RawTerms
def group : MergeGroup := .operator 4934 32
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4934) (leftOrdinal := 0)
    (rightResult := 32) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10652⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101329

namespace LeftMerge101334
def owner : Owner := ⟨.program ⟨214⟩, ⟨7110⟩⟩
def mergeEvent : Nat := 101334
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact27RawTerms
def rightRaw : List Term := Proof.Events056.exact14488RawTerms
def group : MergeGroup := .operator 27 14488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27) (leftOrdinal := 0)
    (rightResult := 14488) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101334

namespace LeftMerge101351
def owner : Owner := ⟨.program ⟨214⟩, ⟨10658⟩⟩
def mergeEvent : Nat := 101351
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101345RawTerms
def rightRaw : List Term := Proof.Events019.exact4937RawTerms
def group : MergeGroup := .operator 101345 4937
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101345) (leftOrdinal := 1)
    (rightResult := 4937) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9490⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩, ⟨.program ⟨214⟩, ⟨10652⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge101351

namespace LeftMerge101352
def owner : Owner := ⟨.program ⟨214⟩, ⟨10658⟩⟩
def mergeEvent : Nat := 101352
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }
def leftRaw : List Term := Proof.Events395.exact101345RawTerms
def rightRaw : List Term := Proof.Events019.exact4937RawTerms
def group : MergeGroup := .operator 101345 4937
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 101345) (leftOrdinal := 0)
    (rightResult := 4937) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6773⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9490⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9490⟩⟩], [⟨.program ⟨214⟩, ⟨6773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge101352

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
