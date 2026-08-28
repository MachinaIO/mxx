import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge65349
def owner : Owner := ⟨.program ⟨214⟩, ⟨10344⟩⟩
def mergeEvent : Nat := 65349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65343RawTerms
def rightRaw : List Term := Proof.Events025.exact6487RawTerms
def group : MergeGroup := .operator 65343 6487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65343) (leftOrdinal := 1)
    (rightResult := 6487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65349

namespace LeftMerge65351
def owner : Owner := ⟨.program ⟨214⟩, ⟨10344⟩⟩
def mergeEvent : Nat := 65351
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6457RawTerms
def group : MergeGroup := .relation 65350
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65350) (rhsResult := 6457)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7882⟩⟩) ⟨6790⟩ 6457) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], [⟨.program ⟨214⟩, ⟨6790⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65351

namespace LeftMerge65352
def owner : Owner := ⟨.program ⟨214⟩, ⟨10344⟩⟩
def mergeEvent : Nat := 65352
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65343RawTerms
def rightRaw : List Term := Proof.Events025.exact6487RawTerms
def group : MergeGroup := .operator 65343 6487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65343) (leftOrdinal := 0)
    (rightResult := 6487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65352

namespace LeftMerge65357
def owner : Owner := ⟨.program ⟨214⟩, ⟨13349⟩⟩
def mergeEvent : Nat := 65357
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65353RawTerms
def rightRaw : List Term := Proof.Events255.exact65323RawTerms
def group : MergeGroup := .operator 65353 65323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65353) (leftOrdinal := 1)
    (rightResult := 65323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6790⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩], [⟨.program ⟨214⟩, ⟨6790⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65357

namespace LeftMerge65365
def owner : Owner := ⟨.program ⟨214⟩, ⟨25754⟩⟩
def mergeEvent : Nat := 65365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65359RawTerms
def rightRaw : List Term := Proof.Events255.exact65290RawTerms
def group : MergeGroup := .operator 65359 65290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65359) (leftOrdinal := 1)
    (rightResult := 65290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25753⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65365

namespace LeftMerge65367
def owner : Owner := ⟨.program ⟨214⟩, ⟨25754⟩⟩
def mergeEvent : Nat := 65367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23414⟩⟩] } }
def rhsRaw : List Term := Proof.Events255.exact65287RawTerms
def group : MergeGroup := .relation 65366
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65366) (rhsResult := 65287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25753⟩⟩) ⟨23414⟩ 65287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23414⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], [⟨.program ⟨214⟩, ⟨23414⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65367

namespace LeftMerge65368
def owner : Owner := ⟨.program ⟨214⟩, ⟨25754⟩⟩
def mergeEvent : Nat := 65368
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65359RawTerms
def rightRaw : List Term := Proof.Events255.exact65290RawTerms
def group : MergeGroup := .operator 65359 65290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65359) (leftOrdinal := 0)
    (rightResult := 65290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25753⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65368

namespace LeftMerge65380
def owner : Owner := ⟨.program ⟨214⟩, ⟨5534⟩⟩
def mergeEvent : Nat := 65380
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65165RawTerms
def rightRaw : List Term := Proof.Events025.exact6550RawTerms
def group : MergeGroup := .operator 65165 6550
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65165) (leftOrdinal := 0)
    (rightResult := 6550) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65380

namespace LeftMerge65393
def owner : Owner := ⟨.program ⟨214⟩, ⟨20247⟩⟩
def mergeEvent : Nat := 65393
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20244⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events255.exact65376RawTerms
def group : MergeGroup := .operator 65387 65376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 65376) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20244⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20244⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65393

namespace LeftMerge65472
def owner : Owner := ⟨.program ⟨214⟩, ⟨13343⟩⟩
def mergeEvent : Nat := 65472
def frameStart : Nat := 65442
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events255.exact65468RawTerms
def rightRaw : List Term := Proof.Events255.exact65465RawTerms
def group : MergeGroup := .operator 65468 65465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65468) (leftOrdinal := 0)
    (rightResult := 65465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10340⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65472

namespace LeftMerge65502
def owner : Owner := ⟨.program ⟨214⟩, ⟨13444⟩⟩
def mergeEvent : Nat := 65502
def frameStart : Nat := 65442
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65498RawTerms
def rightRaw : List Term := Proof.Events255.exact65496RawTerms
def group : MergeGroup := .operator 65498 65496
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65498) (leftOrdinal := 0)
    (rightResult := 65496) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65502

namespace LeftMerge65525
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def mergeEvent : Nat := 65525
def frameStart : Nat := 65442
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65521RawTerms
def rightRaw : List Term := Proof.Events255.exact65518RawTerms
def group : MergeGroup := .operator 65521 65518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65521) (leftOrdinal := 0)
    (rightResult := 65518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7882⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65525

namespace LeftMerge65534
def owner : Owner := ⟨.program ⟨214⟩, ⟨25756⟩⟩
def mergeEvent : Nat := 65534
def frameStart : Nat := 65442
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65530RawTerms
def rightRaw : List Term := Proof.Events255.exact65487RawTerms
def group : MergeGroup := .operator 65530 65487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65530) (leftOrdinal := 0)
    (rightResult := 65487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25753⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6770⟩⟩, ⟨.program ⟨214⟩, ⟨7882⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65534

namespace LeftMerge65535
def owner : Owner := ⟨.program ⟨214⟩, ⟨25756⟩⟩
def mergeEvent : Nat := 65535
def frameStart : Nat := 65442
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65530RawTerms
def rightRaw : List Term := Proof.Events255.exact65487RawTerms
def group : MergeGroup := .operator 65530 65487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65530) (leftOrdinal := 1)
    (rightResult := 65487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25753⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65535

namespace LeftMerge65537
def owner : Owner := ⟨.program ⟨214⟩, ⟨25756⟩⟩
def mergeEvent : Nat := 65537
def frameStart : Nat := 65442
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23414⟩⟩] } }
def rhsRaw : List Term := Proof.Events255.exact65484RawTerms
def group : MergeGroup := .relation 65536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 65536) (rhsResult := 65484)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25753⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25753⟩⟩) ⟨23414⟩ 65484) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23414⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10340⟩⟩, ⟨.program ⟨214⟩, ⟨13342⟩⟩], [⟨.program ⟨214⟩, ⟨23414⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge65537

namespace LeftMerge65545
def owner : Owner := ⟨.program ⟨214⟩, ⟨17009⟩⟩
def mergeEvent : Nat := 65545
def frameStart : Nat := 65442
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17007⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65498RawTerms
def rightRaw : List Term := Proof.Events256.exact65541RawTerms
def group : MergeGroup := .operator 65498 65541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65498) (leftOrdinal := 0)
    (rightResult := 65541) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17007⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17007⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge65545

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
