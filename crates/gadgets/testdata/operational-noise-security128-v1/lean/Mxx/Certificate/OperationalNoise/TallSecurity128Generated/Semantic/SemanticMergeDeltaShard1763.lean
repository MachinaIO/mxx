import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge285393
def owner : Owner := ⟨.program ⟨257⟩, ⟨64264⟩⟩
def mergeEvent : Nat := 285393
def frameStart : Nat := 285327
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285389RawTerms
def rightRaw : List Term := Proof.Events1114.exact285387RawTerms
def group : MergeGroup := .operator 285389 285387
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285389) (leftOrdinal := 0)
    (rightResult := 285387) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285393

namespace LeftMerge285405
def owner : Owner := ⟨.program ⟨257⟩, ⟨64687⟩⟩
def mergeEvent : Nat := 285405
def frameStart : Nat := 285327
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285401RawTerms
def rightRaw : List Term := Proof.Events1114.exact285378RawTerms
def group : MergeGroup := .operator 285401 285378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285401) (leftOrdinal := 0)
    (rightResult := 285378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64686⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285405

namespace LeftMerge285406
def owner : Owner := ⟨.program ⟨257⟩, ⟨64687⟩⟩
def mergeEvent : Nat := 285406
def frameStart : Nat := 285327
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285401RawTerms
def rightRaw : List Term := Proof.Events1114.exact285378RawTerms
def group : MergeGroup := .operator 285401 285378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285401) (leftOrdinal := 1)
    (rightResult := 285378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64686⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285406

namespace LeftMerge285408
def owner : Owner := ⟨.program ⟨257⟩, ⟨64687⟩⟩
def mergeEvent : Nat := 285408
def frameStart : Nat := 285327
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64027⟩⟩] } }
def rhsRaw : List Term := Proof.Events1114.exact285375RawTerms
def group : MergeGroup := .relation 285407
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285407) (rhsResult := 285375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64686⟩⟩) ⟨64027⟩ 285375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64027⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨64027⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285408

namespace LeftMerge285416
def owner : Owner := ⟨.program ⟨257⟩, ⟨62969⟩⟩
def mergeEvent : Nat := 285416
def frameStart : Nat := 285327
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62967⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285389RawTerms
def rightRaw : List Term := Proof.Events1114.exact285412RawTerms
def group : MergeGroup := .operator 285389 285412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285389) (leftOrdinal := 0)
    (rightResult := 285412) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62967⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285416

namespace LeftMerge285433
def owner : Owner := ⟨.program ⟨257⟩, ⟨63559⟩⟩
def mergeEvent : Nat := 285433
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }
def rhsRaw : List Term := Proof.Events1114.exact285430RawTerms
def group : MergeGroup := .relation 285432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285432) (rhsResult := 285430)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 285431 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩]⟩) (none) 285430) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285433

namespace LeftMerge285434
def owner : Owner := ⟨.program ⟨257⟩, ⟨63559⟩⟩
def mergeEvent : Nat := 285434
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩] } }
def rhsRaw : List Term := Proof.Events1114.exact285430RawTerms
def group : MergeGroup := .relation 285432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285432) (rhsResult := 285430)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 285431 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩]⟩) (none) 285430) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285434

namespace LeftMerge285435
def owner : Owner := ⟨.program ⟨257⟩, ⟨63559⟩⟩
def mergeEvent : Nat := 285435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64027⟩⟩] } }
def rhsRaw : List Term := Proof.Events1114.exact285430RawTerms
def group : MergeGroup := .relation 285432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285432) (rhsResult := 285430)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 285431 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩]⟩) (none) 285430) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64027⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨64027⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285435

namespace LeftMerge285436
def owner : Owner := ⟨.program ⟨257⟩, ⟨63559⟩⟩
def mergeEvent : Nat := 285436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1114.exact285430RawTerms
def group : MergeGroup := .relation 285432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 285432) (rhsResult := 285430)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 285431 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63556⟩⟩]⟩) (none) 285430) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62967⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62967⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285436

namespace LeftMerge285441
def owner : Owner := ⟨.program ⟨257⟩, ⟨64689⟩⟩
def mergeEvent : Nat := 285441
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285437RawTerms
def rightRaw : List Term := Proof.Events1114.exact285259RawTerms
def group : MergeGroup := .operator 285437 285259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285437) (leftOrdinal := 0)
    (rightResult := 285259) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64686⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285441

namespace LeftMerge285442
def owner : Owner := ⟨.program ⟨257⟩, ⟨64689⟩⟩
def mergeEvent : Nat := 285442
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64027⟩⟩] } }
def leftRaw : List Term := Proof.Events1114.exact285437RawTerms
def rightRaw : List Term := Proof.Events1114.exact285259RawTerms
def group : MergeGroup := .operator 285437 285259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285437) (leftOrdinal := 2)
    (rightResult := 285259) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64027⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64027⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨62760⟩⟩], [⟨.program ⟨257⟩, ⟨64027⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285442

namespace LeftMerge285468
def owner : Owner := ⟨.program ⟨257⟩, ⟨25179⟩⟩
def mergeEvent : Nat := 285468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13782RawTerms
def rightRaw : List Term := Proof.Events1096.exact280653RawTerms
def group : MergeGroup := .operator 13782 280653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13782) (leftOrdinal := 0)
    (rightResult := 280653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25178⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285468

namespace LeftMerge285473
def owner : Owner := ⟨.program ⟨257⟩, ⟨7896⟩⟩
def mergeEvent : Nat := 285473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events1095.exact280523RawTerms
def rightRaw : List Term := Proof.Events086.exact22090RawTerms
def group : MergeGroup := .operator 280523 22090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280523) (leftOrdinal := 0)
    (rightResult := 22090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285473

namespace LeftMerge285490
def owner : Owner := ⟨.program ⟨257⟩, ⟨59326⟩⟩
def mergeEvent : Nat := 285490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285484RawTerms
def rightRaw : List Term := Proof.Events053.exact13785RawTerms
def group : MergeGroup := .operator 285484 13785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285484) (leftOrdinal := 1)
    (rightResult := 13785) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨25178⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge285490

namespace LeftMerge285491
def owner : Owner := ⟨.program ⟨257⟩, ⟨59326⟩⟩
def mergeEvent : Nat := 285491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events1115.exact285484RawTerms
def rightRaw : List Term := Proof.Events053.exact13785RawTerms
def group : MergeGroup := .operator 285484 13785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 285484) (leftOrdinal := 0)
    (rightResult := 13785) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285491

namespace LeftMerge285496
def owner : Owner := ⟨.program ⟨257⟩, ⟨59327⟩⟩
def mergeEvent : Nat := 285496
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events053.exact13785RawTerms
def rightRaw : List Term := Proof.Events1096.exact280653RawTerms
def group : MergeGroup := .operator 13785 280653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13785) (leftOrdinal := 0)
    (rightResult := 280653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59323⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨59323⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge285496

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
