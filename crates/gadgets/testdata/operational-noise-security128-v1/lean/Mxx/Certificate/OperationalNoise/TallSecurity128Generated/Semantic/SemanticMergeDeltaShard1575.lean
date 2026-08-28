import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge255297
def owner : Owner := ⟨.program ⟨257⟩, ⟨65313⟩⟩
def mergeEvent : Nat := 255297
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def leftRaw : List Term := Proof.Events997.exact255290RawTerms
def rightRaw : List Term := Proof.Events047.exact12249RawTerms
def group : MergeGroup := .operator 255290 12249
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255290) (leftOrdinal := 0)
    (rightResult := 12249) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255297

namespace LeftMerge255302
def owner : Owner := ⟨.program ⟨257⟩, ⟨65314⟩⟩
def mergeEvent : Nat := 255302
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events047.exact12249RawTerms
def rightRaw : List Term := Proof.Events982.exact251403RawTerms
def group : MergeGroup := .operator 12249 251403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12249) (leftOrdinal := 0)
    (rightResult := 251403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255302

namespace LeftMerge255307
def owner : Owner := ⟨.program ⟨257⟩, ⟨8030⟩⟩
def mergeEvent : Nat := 255307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251273RawTerms
def rightRaw : List Term := Proof.Events082.exact21129RawTerms
def group : MergeGroup := .operator 251273 21129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251273) (leftOrdinal := 0)
    (rightResult := 21129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255307

namespace LeftMerge255324
def owner : Owner := ⟨.program ⟨257⟩, ⟨65317⟩⟩
def mergeEvent : Nat := 255324
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events997.exact255318RawTerms
def rightRaw : List Term := Proof.Events082.exact21118RawTerms
def group : MergeGroup := .operator 255318 21118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255318) (leftOrdinal := 1)
    (rightResult := 21118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255324

namespace LeftMerge255326
def owner : Owner := ⟨.program ⟨257⟩, ⟨65317⟩⟩
def mergeEvent : Nat := 255326
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def rhsRaw : List Term := Proof.Events082.exact21088RawTerms
def group : MergeGroup := .relation 255325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255325) (rhsResult := 21088)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9541⟩⟩) ⟨7276⟩ 21088) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255326

namespace LeftMerge255327
def owner : Owner := ⟨.program ⟨257⟩, ⟨65317⟩⟩
def mergeEvent : Nat := 255327
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events997.exact255318RawTerms
def rightRaw : List Term := Proof.Events082.exact21118RawTerms
def group : MergeGroup := .operator 255318 21118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255318) (leftOrdinal := 0)
    (rightResult := 21118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255327

namespace LeftMerge255332
def owner : Owner := ⟨.program ⟨257⟩, ⟨65318⟩⟩
def mergeEvent : Nat := 255332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def leftRaw : List Term := Proof.Events997.exact255328RawTerms
def rightRaw : List Term := Proof.Events997.exact255298RawTerms
def group : MergeGroup := .operator 255328 255298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255328) (leftOrdinal := 1)
    (rightResult := 255298) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255332

namespace LeftMerge255340
def owner : Owner := ⟨.program ⟨257⟩, ⟨69186⟩⟩
def mergeEvent : Nat := 255340
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩] } }
def leftRaw : List Term := Proof.Events997.exact255334RawTerms
def rightRaw : List Term := Proof.Events997.exact255270RawTerms
def group : MergeGroup := .operator 255334 255270
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255334) (leftOrdinal := 1)
    (rightResult := 255270) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69185⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255340

namespace LeftMerge255342
def owner : Owner := ⟨.program ⟨257⟩, ⟨69186⟩⟩
def mergeEvent : Nat := 255342
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68500⟩⟩] } }
def rhsRaw : List Term := Proof.Events997.exact255267RawTerms
def group : MergeGroup := .relation 255341
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 255341) (rhsResult := 255267)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69185⟩⟩) ⟨68500⟩ 255267) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68500⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨68500⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255342

namespace LeftMerge255343
def owner : Owner := ⟨.program ⟨257⟩, ⟨69186⟩⟩
def mergeEvent : Nat := 255343
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩] } }
def leftRaw : List Term := Proof.Events997.exact255334RawTerms
def rightRaw : List Term := Proof.Events997.exact255270RawTerms
def group : MergeGroup := .operator 255334 255270
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255334) (leftOrdinal := 0)
    (rightResult := 255270) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69185⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255343

namespace LeftMerge255357
def owner : Owner := ⟨.program ⟨257⟩, ⟨67723⟩⟩
def mergeEvent : Nat := 255357
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events997.exact255351RawTerms
def group : MergeGroup := .operator 251495 255351
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 255351) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67720⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67720⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255357

namespace LeftMerge255436
def owner : Owner := ⟨.program ⟨257⟩, ⟨65311⟩⟩
def mergeEvent : Nat := 255436
def frameStart : Nat := 255406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events997.exact255432RawTerms
def rightRaw : List Term := Proof.Events997.exact255429RawTerms
def group : MergeGroup := .operator 255432 255429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255432) (leftOrdinal := 0)
    (rightResult := 255429) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25670⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255436

namespace LeftMerge255466
def owner : Owner := ⟨.program ⟨257⟩, ⟨68909⟩⟩
def mergeEvent : Nat := 255466
def frameStart : Nat := 255406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events997.exact255462RawTerms
def rightRaw : List Term := Proof.Events997.exact255460RawTerms
def group : MergeGroup := .operator 255462 255460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255462) (leftOrdinal := 0)
    (rightResult := 255460) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255466

namespace LeftMerge255489
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def mergeEvent : Nat := 255489
def frameStart : Nat := 255406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events997.exact255485RawTerms
def rightRaw : List Term := Proof.Events997.exact255482RawTerms
def group : MergeGroup := .operator 255485 255482
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255485) (leftOrdinal := 0)
    (rightResult := 255482) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255489

namespace LeftMerge255498
def owner : Owner := ⟨.program ⟨257⟩, ⟨69188⟩⟩
def mergeEvent : Nat := 255498
def frameStart : Nat := 255406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩] } }
def leftRaw : List Term := Proof.Events998.exact255494RawTerms
def rightRaw : List Term := Proof.Events997.exact255451RawTerms
def group : MergeGroup := .operator 255494 255451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255494) (leftOrdinal := 0)
    (rightResult := 255451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69185⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge255498

namespace LeftMerge255499
def owner : Owner := ⟨.program ⟨257⟩, ⟨69188⟩⟩
def mergeEvent : Nat := 255499
def frameStart : Nat := 255406
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩] } }
def leftRaw : List Term := Proof.Events998.exact255494RawTerms
def rightRaw : List Term := Proof.Events997.exact255451RawTerms
def group : MergeGroup := .operator 255494 255451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 255494) (leftOrdinal := 1)
    (rightResult := 255451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69185⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25670⟩⟩, ⟨.program ⟨257⟩, ⟨65310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69185⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge255499

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
