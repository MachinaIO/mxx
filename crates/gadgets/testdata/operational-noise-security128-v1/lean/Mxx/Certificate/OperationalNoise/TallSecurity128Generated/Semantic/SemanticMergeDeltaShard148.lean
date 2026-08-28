import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge28321
def owner : Owner := ⟨.program ⟨257⟩, ⟨44448⟩⟩
def mergeEvent : Nat := 28321
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩] } }
def leftRaw : List Term := Proof.Events110.exact28316RawTerms
def rightRaw : List Term := Proof.Events109.exact28138RawTerms
def group : MergeGroup := .operator 28316 28138
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28316) (leftOrdinal := 0)
    (rightResult := 28138) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44445⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28321

namespace LeftMerge28329
def owner : Owner := ⟨.program ⟨257⟩, ⟨44449⟩⟩
def mergeEvent : Nat := 28329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩] } }
def leftRaw : List Term := Proof.Events110.exact28323RawTerms
def rightRaw : List Term := Proof.Events060.exact15582RawTerms
def group : MergeGroup := .operator 28323 15582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28323) (leftOrdinal := 0)
    (rightResult := 15582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28329

namespace LeftMerge28330
def owner : Owner := ⟨.program ⟨257⟩, ⟨44449⟩⟩
def mergeEvent : Nat := 28330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩] } }
def leftRaw : List Term := Proof.Events110.exact28323RawTerms
def rightRaw : List Term := Proof.Events060.exact15582RawTerms
def group : MergeGroup := .operator 28323 15582
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28323) (leftOrdinal := 1)
    (rightResult := 15582) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28330

namespace LeftMerge28332
def owner : Owner := ⟨.program ⟨257⟩, ⟨44449⟩⟩
def mergeEvent : Nat := 28332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15575RawTerms
def group : MergeGroup := .relation 28331
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28331) (rhsResult := 15575)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7153⟩⟩) ⟨7042⟩ 15575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42888⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28332

namespace LeftMerge28346
def owner : Owner := ⟨.program ⟨257⟩, ⟨41767⟩⟩
def mergeEvent : Nat := 28346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18858RawTerms
def rightRaw : List Term := Proof.Events110.exact28340RawTerms
def group : MergeGroup := .operator 18858 28340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18858) (leftOrdinal := 1)
    (rightResult := 28340) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41765⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28346

namespace LeftMerge28348
def owner : Owner := ⟨.program ⟨257⟩, ⟨41767⟩⟩
def mergeEvent : Nat := 28348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41182⟩⟩] } }
def rhsRaw : List Term := Proof.Events110.exact28337RawTerms
def group : MergeGroup := .relation 28347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28347) (rhsResult := 28337)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41765⟩⟩) ⟨41182⟩ 28337) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28348

namespace LeftMerge28349
def owner : Owner := ⟨.program ⟨257⟩, ⟨41767⟩⟩
def mergeEvent : Nat := 28349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩] } }
def leftRaw : List Term := Proof.Events073.exact18858RawTerms
def rightRaw : List Term := Proof.Events110.exact28340RawTerms
def group : MergeGroup := .operator 18858 28340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 18858) (leftOrdinal := 0)
    (rightResult := 28340) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41765⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28349

namespace LeftMerge28363
def owner : Owner := ⟨.program ⟨257⟩, ⟨40681⟩⟩
def mergeEvent : Nat := 28363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events110.exact28357RawTerms
def group : MergeGroup := .operator 17169 28357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 28357) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40678⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28363

namespace LeftMerge28484
def owner : Owner := ⟨.program ⟨257⟩, ⟨41432⟩⟩
def mergeEvent : Nat := 28484
def frameStart : Nat := 28418
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events111.exact28480RawTerms
def rightRaw : List Term := Proof.Events111.exact28478RawTerms
def group : MergeGroup := .operator 28480 28478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28480) (leftOrdinal := 0)
    (rightResult := 28478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28484

namespace LeftMerge28496
def owner : Owner := ⟨.program ⟨257⟩, ⟨41766⟩⟩
def mergeEvent : Nat := 28496
def frameStart : Nat := 28418
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩] } }
def leftRaw : List Term := Proof.Events111.exact28492RawTerms
def rightRaw : List Term := Proof.Events111.exact28469RawTerms
def group : MergeGroup := .operator 28492 28469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28492) (leftOrdinal := 1)
    (rightResult := 28469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41765⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28496

namespace LeftMerge28498
def owner : Owner := ⟨.program ⟨257⟩, ⟨41766⟩⟩
def mergeEvent : Nat := 28498
def frameStart : Nat := 28418
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41182⟩⟩] } }
def rhsRaw : List Term := Proof.Events111.exact28466RawTerms
def group : MergeGroup := .relation 28497
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28497) (rhsResult := 28466)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41765⟩⟩) ⟨41182⟩ 28466) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28498

namespace LeftMerge28499
def owner : Owner := ⟨.program ⟨257⟩, ⟨41766⟩⟩
def mergeEvent : Nat := 28499
def frameStart : Nat := 28418
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩] } }
def leftRaw : List Term := Proof.Events111.exact28492RawTerms
def rightRaw : List Term := Proof.Events111.exact28469RawTerms
def group : MergeGroup := .operator 28492 28469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28492) (leftOrdinal := 0)
    (rightResult := 28469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41765⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28499

namespace LeftMerge28507
def owner : Owner := ⟨.program ⟨257⟩, ⟨40210⟩⟩
def mergeEvent : Nat := 28507
def frameStart : Nat := 28418
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40208⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events111.exact28480RawTerms
def rightRaw : List Term := Proof.Events111.exact28503RawTerms
def group : MergeGroup := .operator 28480 28503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 28480) (leftOrdinal := 0)
    (rightResult := 28503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40208⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40208⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28507

namespace LeftMerge28524
def owner : Owner := ⟨.program ⟨257⟩, ⟨40681⟩⟩
def mergeEvent : Nat := 28524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }
def rhsRaw : List Term := Proof.Events111.exact28521RawTerms
def group : MergeGroup := .relation 28523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28523) (rhsResult := 28521)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28522 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩) (none) 28521) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28524

namespace LeftMerge28525
def owner : Owner := ⟨.program ⟨257⟩, ⟨40681⟩⟩
def mergeEvent : Nat := 28525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41182⟩⟩] } }
def rhsRaw : List Term := Proof.Events111.exact28521RawTerms
def group : MergeGroup := .relation 28523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28523) (rhsResult := 28521)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28522 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩) (none) 28521) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨40038⟩⟩], [⟨.program ⟨257⟩, ⟨41182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge28525

namespace LeftMerge28526
def owner : Owner := ⟨.program ⟨257⟩, ⟨40681⟩⟩
def mergeEvent : Nat := 28526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩] } }
def rhsRaw : List Term := Proof.Events111.exact28521RawTerms
def group : MergeGroup := .relation 28523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 28523) (rhsResult := 28521)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 28522 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40678⟩⟩]⟩) (none) 28521) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41765⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge28526

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
