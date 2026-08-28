import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge36252
def owner : Owner := ⟨.program ⟨214⟩, ⟨13456⟩⟩
def mergeEvent : Nat := 36252
def frameStart : Nat := 36192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36248RawTerms
def rightRaw : List Term := Proof.Events141.exact36246RawTerms
def group : MergeGroup := .operator 36248 36246
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36248) (leftOrdinal := 0)
    (rightResult := 36246) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36252

namespace LeftMerge36275
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def mergeEvent : Nat := 36275
def frameStart : Nat := 36192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36271RawTerms
def rightRaw : List Term := Proof.Events141.exact36268RawTerms
def group : MergeGroup := .operator 36271 36268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36271) (leftOrdinal := 0)
    (rightResult := 36268) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36275

namespace LeftMerge36284
def owner : Owner := ⟨.program ⟨214⟩, ⟨25771⟩⟩
def mergeEvent : Nat := 36284
def frameStart : Nat := 36192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36280RawTerms
def rightRaw : List Term := Proof.Events141.exact36237RawTerms
def group : MergeGroup := .operator 36280 36237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36280) (leftOrdinal := 0)
    (rightResult := 36237) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25768⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36284

namespace LeftMerge36285
def owner : Owner := ⟨.program ⟨214⟩, ⟨25771⟩⟩
def mergeEvent : Nat := 36285
def frameStart : Nat := 36192
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36280RawTerms
def rightRaw : List Term := Proof.Events141.exact36237RawTerms
def group : MergeGroup := .operator 36280 36237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36280) (leftOrdinal := 1)
    (rightResult := 36237) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25768⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36285

namespace LeftMerge36287
def owner : Owner := ⟨.program ⟨214⟩, ⟨25771⟩⟩
def mergeEvent : Nat := 36287
def frameStart : Nat := 36192
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23420⟩⟩] } }
def rhsRaw : List Term := Proof.Events141.exact36234RawTerms
def group : MergeGroup := .relation 36286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36286) (rhsResult := 36234)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25768⟩⟩) ⟨23420⟩ 36234) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23420⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], [⟨.program ⟨214⟩, ⟨23420⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36287

namespace LeftMerge36295
def owner : Owner := ⟨.program ⟨214⟩, ⟨17021⟩⟩
def mergeEvent : Nat := 36295
def frameStart : Nat := 36192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36248RawTerms
def rightRaw : List Term := Proof.Events141.exact36291RawTerms
def group : MergeGroup := .operator 36248 36291
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36248) (leftOrdinal := 0)
    (rightResult := 36291) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36295

namespace LeftMerge36312
def owner : Owner := ⟨.program ⟨214⟩, ⟨20259⟩⟩
def mergeEvent : Nat := 36312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }
def rhsRaw : List Term := Proof.Events141.exact36309RawTerms
def group : MergeGroup := .relation 36311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36311) (rhsResult := 36309)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36310 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20256⟩⟩]⟩) (none) 36309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36312

namespace LeftMerge36313
def owner : Owner := ⟨.program ⟨214⟩, ⟨20259⟩⟩
def mergeEvent : Nat := 36313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩] } }
def rhsRaw : List Term := Proof.Events141.exact36309RawTerms
def group : MergeGroup := .relation 36311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36311) (rhsResult := 36309)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36310 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20256⟩⟩]⟩) (none) 36309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36313

namespace LeftMerge36314
def owner : Owner := ⟨.program ⟨214⟩, ⟨20259⟩⟩
def mergeEvent : Nat := 36314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23420⟩⟩] } }
def rhsRaw : List Term := Proof.Events141.exact36309RawTerms
def group : MergeGroup := .relation 36311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36311) (rhsResult := 36309)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36310 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20256⟩⟩]⟩) (none) 36309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23420⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], [⟨.program ⟨214⟩, ⟨23420⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36314

namespace LeftMerge36315
def owner : Owner := ⟨.program ⟨214⟩, ⟨20259⟩⟩
def mergeEvent : Nat := 36315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events141.exact36309RawTerms
def group : MergeGroup := .relation 36311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36311) (rhsResult := 36309)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20256⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36310 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20256⟩⟩]⟩) (none) 36309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36315

namespace LeftMerge36320
def owner : Owner := ⟨.program ⟨214⟩, ⟨25770⟩⟩
def mergeEvent : Nat := 36320
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23420⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36316RawTerms
def rightRaw : List Term := Proof.Events141.exact36119RawTerms
def group : MergeGroup := .operator 36316 36119
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36316) (leftOrdinal := 2)
    (rightResult := 36119) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23420⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23420⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨10355⟩⟩, ⟨.program ⟨214⟩, ⟨13366⟩⟩], [⟨.program ⟨214⟩, ⟨23420⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36320

namespace LeftMerge36321
def owner : Owner := ⟨.program ⟨214⟩, ⟨25770⟩⟩
def mergeEvent : Nat := 36321
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36316RawTerms
def rightRaw : List Term := Proof.Events141.exact36119RawTerms
def group : MergeGroup := .operator 36316 36119
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36316) (leftOrdinal := 1)
    (rightResult := 36119) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36321

namespace LeftMerge36329
def owner : Owner := ⟨.program ⟨214⟩, ⟨30163⟩⟩
def mergeEvent : Nat := 36329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30161⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36323RawTerms
def rightRaw : List Term := Proof.Events140.exact36030RawTerms
def group : MergeGroup := .operator 36323 36030
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36323) (leftOrdinal := 0)
    (rightResult := 36030) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30161⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30161⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36329

namespace LeftMerge36330
def owner : Owner := ⟨.program ⟨214⟩, ⟨30163⟩⟩
def mergeEvent : Nat := 36330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30161⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36323RawTerms
def rightRaw : List Term := Proof.Events140.exact36030RawTerms
def group : MergeGroup := .operator 36323 36030
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36323) (leftOrdinal := 1)
    (rightResult := 36030) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨30161⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30161⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36330

namespace LeftMerge36332
def owner : Owner := ⟨.program ⟨214⟩, ⟨30163⟩⟩
def mergeEvent : Nat := 36332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24798⟩⟩] } }
def rhsRaw : List Term := Proof.Events140.exact36027RawTerms
def group : MergeGroup := .relation 36331
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36331) (rhsResult := 36027)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨30161⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨30161⟩⟩) ⟨24798⟩ 36027) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24798⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨24798⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36332

namespace LeftMerge36346
def owner : Owner := ⟨.program ⟨214⟩, ⟨22851⟩⟩
def mergeEvent : Nat := 36346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22848⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events141.exact36340RawTerms
def group : MergeGroup := .operator 36137 36340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 36340) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22848⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22848⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36346

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
