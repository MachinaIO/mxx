import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge242320
def owner : Owner := ⟨.program ⟨257⟩, ⟨58460⟩⟩
def mergeEvent : Nat := 242320
def frameStart : Nat := 242227
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩] } }
def leftRaw : List Term := Proof.Events946.exact242315RawTerms
def rightRaw : List Term := Proof.Events946.exact242272RawTerms
def group : MergeGroup := .operator 242315 242272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242315) (leftOrdinal := 1)
    (rightResult := 242272) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58457⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242320

namespace LeftMerge242322
def owner : Owner := ⟨.program ⟨257⟩, ⟨58460⟩⟩
def mergeEvent : Nat := 242322
def frameStart : Nat := 242227
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57957⟩⟩] } }
def rhsRaw : List Term := Proof.Events946.exact242269RawTerms
def group : MergeGroup := .relation 242321
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242321) (rhsResult := 242269)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58457⟩⟩) ⟨57957⟩ 242269) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57957⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], [⟨.program ⟨257⟩, ⟨57957⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242322

namespace LeftMerge242330
def owner : Owner := ⟨.program ⟨257⟩, ⟨56834⟩⟩
def mergeEvent : Nat := 242330
def frameStart : Nat := 242227
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events946.exact242283RawTerms
def rightRaw : List Term := Proof.Events946.exact242326RawTerms
def group : MergeGroup := .operator 242283 242326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242283) (leftOrdinal := 0)
    (rightResult := 242326) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242330

namespace LeftMerge242347
def owner : Owner := ⟨.program ⟨257⟩, ⟨57392⟩⟩
def mergeEvent : Nat := 242347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events946.exact242344RawTerms
def group : MergeGroup := .relation 242346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242346) (rhsResult := 242344)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57389⟩⟩]⟩) (none) 242344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242347

namespace LeftMerge242348
def owner : Owner := ⟨.program ⟨257⟩, ⟨57392⟩⟩
def mergeEvent : Nat := 242348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩] } }
def rhsRaw : List Term := Proof.Events946.exact242344RawTerms
def group : MergeGroup := .relation 242346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242346) (rhsResult := 242344)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57389⟩⟩]⟩) (none) 242344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242348

namespace LeftMerge242349
def owner : Owner := ⟨.program ⟨257⟩, ⟨57392⟩⟩
def mergeEvent : Nat := 242349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57957⟩⟩] } }
def rhsRaw : List Term := Proof.Events946.exact242344RawTerms
def group : MergeGroup := .relation 242346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242346) (rhsResult := 242344)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57389⟩⟩]⟩) (none) 242344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57957⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], [⟨.program ⟨257⟩, ⟨57957⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242349

namespace LeftMerge242350
def owner : Owner := ⟨.program ⟨257⟩, ⟨57392⟩⟩
def mergeEvent : Nat := 242350
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events946.exact242344RawTerms
def group : MergeGroup := .relation 242346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242346) (rhsResult := 242344)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 242345 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57389⟩⟩]⟩) (none) 242344) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242350

namespace LeftMerge242355
def owner : Owner := ⟨.program ⟨257⟩, ⟨58459⟩⟩
def mergeEvent : Nat := 242355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57957⟩⟩] } }
def leftRaw : List Term := Proof.Events946.exact242351RawTerms
def rightRaw : List Term := Proof.Events945.exact242165RawTerms
def group : MergeGroup := .operator 242351 242165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242351) (leftOrdinal := 2)
    (rightResult := 242165) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57957⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57957⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨24986⟩⟩, ⟨.program ⟨257⟩, ⟨56451⟩⟩], [⟨.program ⟨257⟩, ⟨57957⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242355

namespace LeftMerge242356
def owner : Owner := ⟨.program ⟨257⟩, ⟨58459⟩⟩
def mergeEvent : Nat := 242356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩] } }
def leftRaw : List Term := Proof.Events946.exact242351RawTerms
def rightRaw : List Term := Proof.Events945.exact242165RawTerms
def group : MergeGroup := .operator 242351 242165
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242351) (leftOrdinal := 1)
    (rightResult := 242165) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58457⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242356

namespace LeftMerge242364
def owner : Owner := ⟨.program ⟨257⟩, ⟨58852⟩⟩
def mergeEvent : Nat := 242364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩] } }
def leftRaw : List Term := Proof.Events946.exact242358RawTerms
def rightRaw : List Term := Proof.Events945.exact242081RawTerms
def group : MergeGroup := .operator 242358 242081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242358) (leftOrdinal := 0)
    (rightResult := 242081) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58850⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242364

namespace LeftMerge242365
def owner : Owner := ⟨.program ⟨257⟩, ⟨58852⟩⟩
def mergeEvent : Nat := 242365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩] } }
def leftRaw : List Term := Proof.Events946.exact242358RawTerms
def rightRaw : List Term := Proof.Events945.exact242081RawTerms
def group : MergeGroup := .operator 242358 242081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242358) (leftOrdinal := 1)
    (rightResult := 242081) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58850⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242365

namespace LeftMerge242367
def owner : Owner := ⟨.program ⟨257⟩, ⟨58852⟩⟩
def mergeEvent : Nat := 242367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58103⟩⟩] } }
def rhsRaw : List Term := Proof.Events945.exact242078RawTerms
def group : MergeGroup := .relation 242366
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 242366) (rhsResult := 242078)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58850⟩⟩) ⟨58103⟩ 242078) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58103⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨58103⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242367

namespace LeftMerge242381
def owner : Owner := ⟨.program ⟨257⟩, ⟨57679⟩⟩
def mergeEvent : Nat := 242381
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57676⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events946.exact242375RawTerms
def group : MergeGroup := .operator 236870 242375
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 242375) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57676⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242381

namespace LeftMerge242502
def owner : Owner := ⟨.program ⟨257⟩, ⟨58320⟩⟩
def mergeEvent : Nat := 242502
def frameStart : Nat := 242436
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events947.exact242498RawTerms
def rightRaw : List Term := Proof.Events947.exact242496RawTerms
def group : MergeGroup := .operator 242498 242496
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242498) (leftOrdinal := 0)
    (rightResult := 242496) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242502

namespace LeftMerge242514
def owner : Owner := ⟨.program ⟨257⟩, ⟨58851⟩⟩
def mergeEvent : Nat := 242514
def frameStart : Nat := 242436
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩] } }
def leftRaw : List Term := Proof.Events947.exact242510RawTerms
def rightRaw : List Term := Proof.Events947.exact242487RawTerms
def group : MergeGroup := .operator 242510 242487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242510) (leftOrdinal := 0)
    (rightResult := 242487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58850⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge242514

namespace LeftMerge242515
def owner : Owner := ⟨.program ⟨257⟩, ⟨58851⟩⟩
def mergeEvent : Nat := 242515
def frameStart : Nat := 242436
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩] } }
def leftRaw : List Term := Proof.Events947.exact242510RawTerms
def rightRaw : List Term := Proof.Events947.exact242487RawTerms
def group : MergeGroup := .operator 242510 242487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 242510) (leftOrdinal := 1)
    (rightResult := 242487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56832⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58850⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56832⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58850⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge242515

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
