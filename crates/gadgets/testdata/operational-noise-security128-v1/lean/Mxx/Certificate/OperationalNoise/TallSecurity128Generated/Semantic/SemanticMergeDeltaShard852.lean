import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge140429
def owner : Owner := ⟨.program ⟨257⟩, ⟨55425⟩⟩
def mergeEvent : Nat := 140429
def frameStart : Nat := 140334
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54947⟩⟩] } }
def rhsRaw : List Term := Proof.Events548.exact140376RawTerms
def group : MergeGroup := .relation 140428
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140428) (rhsResult := 140376)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55422⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55422⟩⟩) ⟨54947⟩ 140376) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54947⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], [⟨.program ⟨257⟩, ⟨54947⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140429

namespace LeftMerge140437
def owner : Owner := ⟨.program ⟨257⟩, ⟨53814⟩⟩
def mergeEvent : Nat := 140437
def frameStart : Nat := 140334
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events548.exact140390RawTerms
def rightRaw : List Term := Proof.Events548.exact140433RawTerms
def group : MergeGroup := .operator 140390 140433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140390) (leftOrdinal := 0)
    (rightResult := 140433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140437

namespace LeftMerge140454
def owner : Owner := ⟨.program ⟨257⟩, ⟨54362⟩⟩
def mergeEvent : Nat := 140454
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events548.exact140451RawTerms
def group : MergeGroup := .relation 140453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140453) (rhsResult := 140451)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 140452 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54359⟩⟩]⟩) (none) 140451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140454

namespace LeftMerge140455
def owner : Owner := ⟨.program ⟨257⟩, ⟨54362⟩⟩
def mergeEvent : Nat := 140455
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55422⟩⟩] } }
def rhsRaw : List Term := Proof.Events548.exact140451RawTerms
def group : MergeGroup := .relation 140453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140453) (rhsResult := 140451)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 140452 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54359⟩⟩]⟩) (none) 140451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55422⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55422⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140455

namespace LeftMerge140456
def owner : Owner := ⟨.program ⟨257⟩, ⟨54362⟩⟩
def mergeEvent : Nat := 140456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54947⟩⟩] } }
def rhsRaw : List Term := Proof.Events548.exact140451RawTerms
def group : MergeGroup := .relation 140453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140453) (rhsResult := 140451)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 140452 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54359⟩⟩]⟩) (none) 140451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54947⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], [⟨.program ⟨257⟩, ⟨54947⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140456

namespace LeftMerge140457
def owner : Owner := ⟨.program ⟨257⟩, ⟨54362⟩⟩
def mergeEvent : Nat := 140457
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events548.exact140451RawTerms
def group : MergeGroup := .relation 140453
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140453) (rhsResult := 140451)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 140452 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54359⟩⟩]⟩) (none) 140451) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨53812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140457

namespace LeftMerge140462
def owner : Owner := ⟨.program ⟨257⟩, ⟨55424⟩⟩
def mergeEvent : Nat := 140462
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54947⟩⟩] } }
def leftRaw : List Term := Proof.Events548.exact140458RawTerms
def rightRaw : List Term := Proof.Events547.exact140272RawTerms
def group : MergeGroup := .operator 140458 140272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140458) (leftOrdinal := 2)
    (rightResult := 140272) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54947⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54947⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨24686⟩⟩, ⟨.program ⟨257⟩, ⟨53336⟩⟩], [⟨.program ⟨257⟩, ⟨54947⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140462

namespace LeftMerge140463
def owner : Owner := ⟨.program ⟨257⟩, ⟨55424⟩⟩
def mergeEvent : Nat := 140463
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55422⟩⟩] } }
def leftRaw : List Term := Proof.Events548.exact140458RawTerms
def rightRaw : List Term := Proof.Events547.exact140272RawTerms
def group : MergeGroup := .operator 140458 140272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140458) (leftOrdinal := 1)
    (rightResult := 140272) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55422⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55422⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55422⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140463

namespace LeftMerge140471
def owner : Owner := ⟨.program ⟨257⟩, ⟨55717⟩⟩
def mergeEvent : Nat := 140471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩] } }
def leftRaw : List Term := Proof.Events548.exact140465RawTerms
def rightRaw : List Term := Proof.Events547.exact140188RawTerms
def group : MergeGroup := .operator 140465 140188
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140465) (leftOrdinal := 0)
    (rightResult := 140188) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55715⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140471

namespace LeftMerge140472
def owner : Owner := ⟨.program ⟨257⟩, ⟨55717⟩⟩
def mergeEvent : Nat := 140472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩] } }
def leftRaw : List Term := Proof.Events548.exact140465RawTerms
def rightRaw : List Term := Proof.Events547.exact140188RawTerms
def group : MergeGroup := .operator 140465 140188
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140465) (leftOrdinal := 1)
    (rightResult := 140188) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55715⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨53812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140472

namespace LeftMerge140474
def owner : Owner := ⟨.program ⟨257⟩, ⟨55717⟩⟩
def mergeEvent : Nat := 140474
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55078⟩⟩] } }
def rhsRaw : List Term := Proof.Events547.exact140185RawTerms
def group : MergeGroup := .relation 140473
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140473) (rhsResult := 140185)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨53812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55715⟩⟩) ⟨55078⟩ 140185) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55078⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨53812⟩⟩], [⟨.program ⟨257⟩, ⟨55078⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140474

namespace LeftMerge140488
def owner : Owner := ⟨.program ⟨257⟩, ⟨54599⟩⟩
def mergeEvent : Nat := 140488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54596⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events548.exact140482RawTerms
def group : MergeGroup := .operator 134495 140482
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 140482) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54596⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54596⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140488

namespace LeftMerge140609
def owner : Owner := ⟨.program ⟨257⟩, ⟨55320⟩⟩
def mergeEvent : Nat := 140609
def frameStart : Nat := 140543
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events549.exact140605RawTerms
def rightRaw : List Term := Proof.Events549.exact140603RawTerms
def group : MergeGroup := .operator 140605 140603
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140605) (leftOrdinal := 0)
    (rightResult := 140603) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140609

namespace LeftMerge140621
def owner : Owner := ⟨.program ⟨257⟩, ⟨55716⟩⟩
def mergeEvent : Nat := 140621
def frameStart : Nat := 140543
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩] } }
def leftRaw : List Term := Proof.Events549.exact140617RawTerms
def rightRaw : List Term := Proof.Events549.exact140594RawTerms
def group : MergeGroup := .operator 140617 140594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140617) (leftOrdinal := 0)
    (rightResult := 140594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55715⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge140621

namespace LeftMerge140622
def owner : Owner := ⟨.program ⟨257⟩, ⟨55716⟩⟩
def mergeEvent : Nat := 140622
def frameStart : Nat := 140543
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩] } }
def leftRaw : List Term := Proof.Events549.exact140617RawTerms
def rightRaw : List Term := Proof.Events549.exact140594RawTerms
def group : MergeGroup := .operator 140617 140594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 140617) (leftOrdinal := 1)
    (rightResult := 140594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55715⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140622

namespace LeftMerge140624
def owner : Owner := ⟨.program ⟨257⟩, ⟨55716⟩⟩
def mergeEvent : Nat := 140624
def frameStart : Nat := 140543
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55078⟩⟩] } }
def rhsRaw : List Term := Proof.Events549.exact140591RawTerms
def group : MergeGroup := .relation 140623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 140623) (rhsResult := 140591)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55715⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55715⟩⟩) ⟨55078⟩ 140591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55078⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53812⟩⟩], [⟨.program ⟨257⟩, ⟨55078⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge140624

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
