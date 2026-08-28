import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge29216
def owner : Owner := ⟨.program ⟨214⟩, ⟨25004⟩⟩
def mergeEvent : Nat := 29216
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29207RawTerms
def rightRaw : List Term := Proof.Events113.exact29143RawTerms
def group : MergeGroup := .operator 29207 29143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29207) (leftOrdinal := 0)
    (rightResult := 29143) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25003⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29216

namespace LeftMerge29230
def owner : Owner := ⟨.program ⟨214⟩, ⟨19111⟩⟩
def mergeEvent : Nat := 29230
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events114.exact29224RawTerms
def group : MergeGroup := .operator 21512 29224
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 29224) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19108⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29230

namespace LeftMerge29309
def owner : Owner := ⟨.program ⟨214⟩, ⟨10701⟩⟩
def mergeEvent : Nat := 29309
def frameStart : Nat := 29279
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events114.exact29305RawTerms
def rightRaw : List Term := Proof.Events114.exact29302RawTerms
def group : MergeGroup := .operator 29305 29302
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29305) (leftOrdinal := 0)
    (rightResult := 29302) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29309

namespace LeftMerge29339
def owner : Owner := ⟨.program ⟨214⟩, ⟨10786⟩⟩
def mergeEvent : Nat := 29339
def frameStart : Nat := 29279
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29335RawTerms
def rightRaw : List Term := Proof.Events114.exact29333RawTerms
def group : MergeGroup := .operator 29335 29333
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29335) (leftOrdinal := 0)
    (rightResult := 29333) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29339

namespace LeftMerge29362
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def mergeEvent : Nat := 29362
def frameStart : Nat := 29279
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29358RawTerms
def rightRaw : List Term := Proof.Events114.exact29355RawTerms
def group : MergeGroup := .operator 29358 29355
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29358) (leftOrdinal := 0)
    (rightResult := 29355) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7834⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29362

namespace LeftMerge29371
def owner : Owner := ⟨.program ⟨214⟩, ⟨25006⟩⟩
def mergeEvent : Nat := 29371
def frameStart : Nat := 29279
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29367RawTerms
def rightRaw : List Term := Proof.Events114.exact29324RawTerms
def group : MergeGroup := .operator 29367 29324
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29367) (leftOrdinal := 0)
    (rightResult := 29324) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25003⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29371

namespace LeftMerge29372
def owner : Owner := ⟨.program ⟨214⟩, ⟨25006⟩⟩
def mergeEvent : Nat := 29372
def frameStart : Nat := 29279
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29367RawTerms
def rightRaw : List Term := Proof.Events114.exact29324RawTerms
def group : MergeGroup := .operator 29367 29324
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29367) (leftOrdinal := 1)
    (rightResult := 29324) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25003⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29372

namespace LeftMerge29374
def owner : Owner := ⟨.program ⟨214⟩, ⟨25006⟩⟩
def mergeEvent : Nat := 29374
def frameStart : Nat := 29279
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23002⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29321RawTerms
def group : MergeGroup := .relation 29373
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29373) (rhsResult := 29321)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25003⟩⟩) ⟨23002⟩ 29321) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23002⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], [⟨.program ⟨214⟩, ⟨23002⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29374

namespace LeftMerge29382
def owner : Owner := ⟨.program ⟨214⟩, ⟨14967⟩⟩
def mergeEvent : Nat := 29382
def frameStart : Nat := 29279
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29335RawTerms
def rightRaw : List Term := Proof.Events114.exact29378RawTerms
def group : MergeGroup := .operator 29335 29378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29335) (leftOrdinal := 0)
    (rightResult := 29378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29382

namespace LeftMerge29399
def owner : Owner := ⟨.program ⟨214⟩, ⟨19111⟩⟩
def mergeEvent : Nat := 29399
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29396RawTerms
def group : MergeGroup := .relation 29398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29398) (rhsResult := 29396)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29397 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩) (none) 29396) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29399

namespace LeftMerge29400
def owner : Owner := ⟨.program ⟨214⟩, ⟨19111⟩⟩
def mergeEvent : Nat := 29400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29396RawTerms
def group : MergeGroup := .relation 29398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29398) (rhsResult := 29396)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29397 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩) (none) 29396) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29400

namespace LeftMerge29401
def owner : Owner := ⟨.program ⟨214⟩, ⟨19111⟩⟩
def mergeEvent : Nat := 29401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23002⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29396RawTerms
def group : MergeGroup := .relation 29398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29398) (rhsResult := 29396)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29397 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩) (none) 29396) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23002⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], [⟨.program ⟨214⟩, ⟨23002⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29401

namespace LeftMerge29402
def owner : Owner := ⟨.program ⟨214⟩, ⟨19111⟩⟩
def mergeEvent : Nat := 29402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events114.exact29396RawTerms
def group : MergeGroup := .relation 29398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29398) (rhsResult := 29396)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29397 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19108⟩⟩]⟩) (none) 29396) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29402

namespace LeftMerge29407
def owner : Owner := ⟨.program ⟨214⟩, ⟨25005⟩⟩
def mergeEvent : Nat := 29407
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23002⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29403RawTerms
def rightRaw : List Term := Proof.Events114.exact29217RawTerms
def group : MergeGroup := .operator 29403 29217
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29403) (leftOrdinal := 2)
    (rightResult := 29217) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23002⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23002⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9520⟩⟩, ⟨.program ⟨214⟩, ⟨10700⟩⟩], [⟨.program ⟨214⟩, ⟨23002⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29407

namespace LeftMerge29408
def owner : Owner := ⟨.program ⟨214⟩, ⟨25005⟩⟩
def mergeEvent : Nat := 29408
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29403RawTerms
def rightRaw : List Term := Proof.Events114.exact29217RawTerms
def group : MergeGroup := .operator 29403 29217
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29403) (leftOrdinal := 1)
    (rightResult := 29217) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25003⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29408

namespace LeftMerge29416
def owner : Owner := ⟨.program ⟨214⟩, ⟨26605⟩⟩
def mergeEvent : Nat := 29416
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29410RawTerms
def rightRaw : List Term := Proof.Events113.exact29133RawTerms
def group : MergeGroup := .operator 29410 29133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29410) (leftOrdinal := 0)
    (rightResult := 29133) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26603⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29416

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
