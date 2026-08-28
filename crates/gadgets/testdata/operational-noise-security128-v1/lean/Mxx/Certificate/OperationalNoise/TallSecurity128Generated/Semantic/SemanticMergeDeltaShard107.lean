import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge21327
def owner : Owner := ⟨.program ⟨257⟩, ⟨69147⟩⟩
def mergeEvent : Nat := 21327
def frameStart : Nat := 21232
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69144⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21320RawTerms
def rightRaw : List Term := Proof.Events083.exact21277RawTerms
def group : MergeGroup := .operator 21320 21277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21320) (leftOrdinal := 0)
    (rightResult := 21277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69144⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69144⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21327

namespace LeftMerge21335
def owner : Owner := ⟨.program ⟨257⟩, ⟨65720⟩⟩
def mergeEvent : Nat := 21335
def frameStart : Nat := 21232
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21288RawTerms
def rightRaw : List Term := Proof.Events083.exact21331RawTerms
def group : MergeGroup := .operator 21288 21331
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21288) (leftOrdinal := 0)
    (rightResult := 21331) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21335

namespace LeftMerge21352
def owner : Owner := ⟨.program ⟨257⟩, ⟨67686⟩⟩
def mergeEvent : Nat := 21352
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25626⟩⟩, ⟨.program ⟨257⟩, ⟨65211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68478⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21349RawTerms
def group : MergeGroup := .relation 21351
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21351) (rhsResult := 21349)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67683⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21350 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67683⟩⟩]⟩) (none) 21349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25626⟩⟩, ⟨.program ⟨257⟩, ⟨65211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68478⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25626⟩⟩, ⟨.program ⟨257⟩, ⟨65211⟩⟩], [⟨.program ⟨257⟩, ⟨68478⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21352

namespace LeftMerge21353
def owner : Owner := ⟨.program ⟨257⟩, ⟨67686⟩⟩
def mergeEvent : Nat := 21353
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69144⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21349RawTerms
def group : MergeGroup := .relation 21351
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21351) (rhsResult := 21349)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67683⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21350 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67683⟩⟩]⟩) (none) 21349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69144⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69144⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21353

namespace LeftMerge21354
def owner : Owner := ⟨.program ⟨257⟩, ⟨67686⟩⟩
def mergeEvent : Nat := 21354
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21349RawTerms
def group : MergeGroup := .relation 21351
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21351) (rhsResult := 21349)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67683⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21350 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67683⟩⟩]⟩) (none) 21349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21354

namespace LeftMerge21355
def owner : Owner := ⟨.program ⟨257⟩, ⟨67686⟩⟩
def mergeEvent : Nat := 21355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21349RawTerms
def group : MergeGroup := .relation 21351
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21351) (rhsResult := 21349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67683⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 21350 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67683⟩⟩]⟩) (none) 21349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21355

namespace LeftMerge21360
def owner : Owner := ⟨.program ⟨257⟩, ⟨69146⟩⟩
def mergeEvent : Nat := 21360
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25626⟩⟩, ⟨.program ⟨257⟩, ⟨65211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68478⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21356RawTerms
def rightRaw : List Term := Proof.Events082.exact21170RawTerms
def group : MergeGroup := .operator 21356 21170
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21356) (leftOrdinal := 2)
    (rightResult := 21170) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25626⟩⟩, ⟨.program ⟨257⟩, ⟨65211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68478⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25626⟩⟩, ⟨.program ⟨257⟩, ⟨65211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68478⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25626⟩⟩, ⟨.program ⟨257⟩, ⟨65211⟩⟩], [⟨.program ⟨257⟩, ⟨68478⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21360

namespace LeftMerge21361
def owner : Owner := ⟨.program ⟨257⟩, ⟨69146⟩⟩
def mergeEvent : Nat := 21361
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69144⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21356RawTerms
def rightRaw : List Term := Proof.Events082.exact21170RawTerms
def group : MergeGroup := .operator 21356 21170
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21356) (leftOrdinal := 1)
    (rightResult := 21170) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69144⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69144⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69144⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21361

namespace LeftMerge21369
def owner : Owner := ⟨.program ⟨257⟩, ⟨69493⟩⟩
def mergeEvent : Nat := 21369
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21363RawTerms
def rightRaw : List Term := Proof.Events082.exact21067RawTerms
def group : MergeGroup := .operator 21363 21067
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21363) (leftOrdinal := 1)
    (rightResult := 21067) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69491⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21369

namespace LeftMerge21371
def owner : Owner := ⟨.program ⟨257⟩, ⟨69493⟩⟩
def mergeEvent : Nat := 21371
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68604⟩⟩] } }
def rhsRaw : List Term := Proof.Events082.exact21064RawTerms
def group : MergeGroup := .relation 21370
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21370) (rhsResult := 21064)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69491⟩⟩) ⟨68604⟩ 21064) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68604⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨68604⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21371

namespace LeftMerge21372
def owner : Owner := ⟨.program ⟨257⟩, ⟨69493⟩⟩
def mergeEvent : Nat := 21372
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21363RawTerms
def rightRaw : List Term := Proof.Events082.exact21067RawTerms
def group : MergeGroup := .operator 21363 21067
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21363) (leftOrdinal := 0)
    (rightResult := 21067) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69491⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21372

namespace LeftMerge21386
def owner : Owner := ⟨.program ⟨257⟩, ⟨67906⟩⟩
def mergeEvent : Nat := 21386
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67903⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events083.exact21380RawTerms
def group : MergeGroup := .operator 17169 21380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 21380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67903⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67903⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21386

namespace LeftMerge21507
def owner : Owner := ⟨.program ⟨257⟩, ⟨68973⟩⟩
def mergeEvent : Nat := 21507
def frameStart : Nat := 21441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21503RawTerms
def rightRaw : List Term := Proof.Events083.exact21501RawTerms
def group : MergeGroup := .operator 21503 21501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21503) (leftOrdinal := 0)
    (rightResult := 21501) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21507

namespace LeftMerge21519
def owner : Owner := ⟨.program ⟨257⟩, ⟨69492⟩⟩
def mergeEvent : Nat := 21519
def frameStart : Nat := 21441
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21515RawTerms
def rightRaw : List Term := Proof.Events083.exact21492RawTerms
def group : MergeGroup := .operator 21515 21492
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21515) (leftOrdinal := 1)
    (rightResult := 21492) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69491⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21519

namespace LeftMerge21521
def owner : Owner := ⟨.program ⟨257⟩, ⟨69492⟩⟩
def mergeEvent : Nat := 21521
def frameStart : Nat := 21441
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65718⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68604⟩⟩] } }
def rhsRaw : List Term := Proof.Events083.exact21489RawTerms
def group : MergeGroup := .relation 21520
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21520) (rhsResult := 21489)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69491⟩⟩) ⟨68604⟩ 21489) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68604⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65718⟩⟩], [⟨.program ⟨257⟩, ⟨68604⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21521

namespace LeftMerge21522
def owner : Owner := ⟨.program ⟨257⟩, ⟨69492⟩⟩
def mergeEvent : Nat := 21522
def frameStart : Nat := 21441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21515RawTerms
def rightRaw : List Term := Proof.Events083.exact21492RawTerms
def group : MergeGroup := .operator 21515 21492
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21515) (leftOrdinal := 0)
    (rightResult := 21492) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69491⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69491⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21522

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
