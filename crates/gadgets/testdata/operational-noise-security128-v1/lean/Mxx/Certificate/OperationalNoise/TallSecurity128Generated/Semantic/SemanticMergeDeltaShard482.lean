import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge81444
def owner : Owner := ⟨.program ⟨257⟩, ⟨58548⟩⟩
def mergeEvent : Nat := 81444
def frameStart : Nat := 81352
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81440RawTerms
def rightRaw : List Term := Proof.Events317.exact81397RawTerms
def group : MergeGroup := .operator 81440 81397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81440) (leftOrdinal := 0)
    (rightResult := 81397) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58545⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81444

namespace LeftMerge81445
def owner : Owner := ⟨.program ⟨257⟩, ⟨58548⟩⟩
def mergeEvent : Nat := 81445
def frameStart : Nat := 81352
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81440RawTerms
def rightRaw : List Term := Proof.Events317.exact81397RawTerms
def group : MergeGroup := .operator 81440 81397
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81440) (leftOrdinal := 1)
    (rightResult := 81397) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58545⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81445

namespace LeftMerge81447
def owner : Owner := ⟨.program ⟨257⟩, ⟨58548⟩⟩
def mergeEvent : Nat := 81447
def frameStart : Nat := 81352
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58005⟩⟩] } }
def rhsRaw : List Term := Proof.Events317.exact81394RawTerms
def group : MergeGroup := .relation 81446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81446) (rhsResult := 81394)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58545⟩⟩) ⟨58005⟩ 81394) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58005⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], [⟨.program ⟨257⟩, ⟨58005⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81447

namespace LeftMerge81455
def owner : Owner := ⟨.program ⟨257⟩, ⟨56898⟩⟩
def mergeEvent : Nat := 81455
def frameStart : Nat := 81352
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81408RawTerms
def rightRaw : List Term := Proof.Events318.exact81451RawTerms
def group : MergeGroup := .operator 81408 81451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81408) (leftOrdinal := 0)
    (rightResult := 81451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56896⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81455

namespace LeftMerge81472
def owner : Owner := ⟨.program ⟨257⟩, ⟨57472⟩⟩
def mergeEvent : Nat := 81472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events318.exact81469RawTerms
def group : MergeGroup := .relation 81471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81471) (rhsResult := 81469)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81470 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57469⟩⟩]⟩) (none) 81469) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81472

namespace LeftMerge81473
def owner : Owner := ⟨.program ⟨257⟩, ⟨57472⟩⟩
def mergeEvent : Nat := 81473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩] } }
def rhsRaw : List Term := Proof.Events318.exact81469RawTerms
def group : MergeGroup := .relation 81471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81471) (rhsResult := 81469)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81470 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57469⟩⟩]⟩) (none) 81469) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81473

namespace LeftMerge81474
def owner : Owner := ⟨.program ⟨257⟩, ⟨57472⟩⟩
def mergeEvent : Nat := 81474
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58005⟩⟩] } }
def rhsRaw : List Term := Proof.Events318.exact81469RawTerms
def group : MergeGroup := .relation 81471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81471) (rhsResult := 81469)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81470 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57469⟩⟩]⟩) (none) 81469) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58005⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], [⟨.program ⟨257⟩, ⟨58005⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81474

namespace LeftMerge81475
def owner : Owner := ⟨.program ⟨257⟩, ⟨57472⟩⟩
def mergeEvent : Nat := 81475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events318.exact81469RawTerms
def group : MergeGroup := .relation 81471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81471) (rhsResult := 81469)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81470 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57469⟩⟩]⟩) (none) 81469) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81475

namespace LeftMerge81480
def owner : Owner := ⟨.program ⟨257⟩, ⟨58547⟩⟩
def mergeEvent : Nat := 81480
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58005⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81476RawTerms
def rightRaw : List Term := Proof.Events317.exact81290RawTerms
def group : MergeGroup := .operator 81476 81290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81476) (leftOrdinal := 2)
    (rightResult := 81290) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58005⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58005⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨25082⟩⟩, ⟨.program ⟨257⟩, ⟨56667⟩⟩], [⟨.program ⟨257⟩, ⟨58005⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81480

namespace LeftMerge81481
def owner : Owner := ⟨.program ⟨257⟩, ⟨58547⟩⟩
def mergeEvent : Nat := 81481
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81476RawTerms
def rightRaw : List Term := Proof.Events317.exact81290RawTerms
def group : MergeGroup := .operator 81476 81290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81476) (leftOrdinal := 1)
    (rightResult := 81290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58545⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81481

namespace LeftMerge81489
def owner : Owner := ⟨.program ⟨257⟩, ⟨59100⟩⟩
def mergeEvent : Nat := 81489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59098⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81483RawTerms
def rightRaw : List Term := Proof.Events317.exact81206RawTerms
def group : MergeGroup := .operator 81483 81206
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81483) (leftOrdinal := 0)
    (rightResult := 81206) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59098⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59098⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81489

namespace LeftMerge81490
def owner : Owner := ⟨.program ⟨257⟩, ⟨59100⟩⟩
def mergeEvent : Nat := 81490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59098⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81483RawTerms
def rightRaw : List Term := Proof.Events317.exact81206RawTerms
def group : MergeGroup := .operator 81483 81206
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81483) (leftOrdinal := 1)
    (rightResult := 81206) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59098⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59098⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81490

namespace LeftMerge81492
def owner : Owner := ⟨.program ⟨257⟩, ⟨59100⟩⟩
def mergeEvent : Nat := 81492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58175⟩⟩] } }
def rhsRaw : List Term := Proof.Events317.exact81203RawTerms
def group : MergeGroup := .relation 81491
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81491) (rhsResult := 81203)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59098⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59098⟩⟩) ⟨58175⟩ 81203) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58175⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨58175⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81492

namespace LeftMerge81506
def owner : Owner := ⟨.program ⟨257⟩, ⟨57839⟩⟩
def mergeEvent : Nat := 81506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57836⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events318.exact81500RawTerms
def group : MergeGroup := .operator 75995 81500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 81500) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57836⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81506

namespace LeftMerge81627
def owner : Owner := ⟨.program ⟨257⟩, ⟨58352⟩⟩
def mergeEvent : Nat := 81627
def frameStart : Nat := 81561
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81623RawTerms
def rightRaw : List Term := Proof.Events318.exact81621RawTerms
def group : MergeGroup := .operator 81623 81621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81623) (leftOrdinal := 0)
    (rightResult := 81621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56896⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56896⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81627

namespace LeftMerge81639
def owner : Owner := ⟨.program ⟨257⟩, ⟨59099⟩⟩
def mergeEvent : Nat := 81639
def frameStart : Nat := 81561
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59098⟩⟩] } }
def leftRaw : List Term := Proof.Events318.exact81635RawTerms
def rightRaw : List Term := Proof.Events318.exact81612RawTerms
def group : MergeGroup := .operator 81635 81612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81635) (leftOrdinal := 0)
    (rightResult := 81612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59098⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59098⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81639

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
