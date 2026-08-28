import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge128273
def owner : Owner := ⟨.program ⟨257⟩, ⟨16519⟩⟩
def mergeEvent : Nat := 128273
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events501.exact128267RawTerms
def group : MergeGroup := .operator 119870 128267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 128267) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16516⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128273

namespace LeftMerge128394
def owner : Owner := ⟨.program ⟨257⟩, ⟨17192⟩⟩
def mergeEvent : Nat := 128394
def frameStart : Nat := 128328
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events501.exact128390RawTerms
def rightRaw : List Term := Proof.Events501.exact128388RawTerms
def group : MergeGroup := .operator 128390 128388
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128390) (leftOrdinal := 0)
    (rightResult := 128388) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128394

namespace LeftMerge128406
def owner : Owner := ⟨.program ⟨257⟩, ⟨17650⟩⟩
def mergeEvent : Nat := 128406
def frameStart : Nat := 128328
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩] } }
def leftRaw : List Term := Proof.Events501.exact128402RawTerms
def rightRaw : List Term := Proof.Events501.exact128379RawTerms
def group : MergeGroup := .operator 128402 128379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128402) (leftOrdinal := 0)
    (rightResult := 128379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17649⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128406

namespace LeftMerge128407
def owner : Owner := ⟨.program ⟨257⟩, ⟨17650⟩⟩
def mergeEvent : Nat := 128407
def frameStart : Nat := 128328
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩] } }
def leftRaw : List Term := Proof.Events501.exact128402RawTerms
def rightRaw : List Term := Proof.Events501.exact128379RawTerms
def group : MergeGroup := .operator 128402 128379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128402) (leftOrdinal := 1)
    (rightResult := 128379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17649⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128407

namespace LeftMerge128409
def owner : Owner := ⟨.program ⟨257⟩, ⟨17650⟩⟩
def mergeEvent : Nat := 128409
def frameStart : Nat := 128328
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16965⟩⟩] } }
def rhsRaw : List Term := Proof.Events501.exact128376RawTerms
def group : MergeGroup := .relation 128408
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128408) (rhsResult := 128376)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17649⟩⟩) ⟨16965⟩ 128376) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16965⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128409

namespace LeftMerge128417
def owner : Owner := ⟨.program ⟨257⟩, ⟨15972⟩⟩
def mergeEvent : Nat := 128417
def frameStart : Nat := 128328
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events501.exact128390RawTerms
def rightRaw : List Term := Proof.Events501.exact128413RawTerms
def group : MergeGroup := .operator 128390 128413
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128390) (leftOrdinal := 0)
    (rightResult := 128413) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128417

namespace LeftMerge128434
def owner : Owner := ⟨.program ⟨257⟩, ⟨16519⟩⟩
def mergeEvent : Nat := 128434
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }
def rhsRaw : List Term := Proof.Events501.exact128431RawTerms
def group : MergeGroup := .relation 128433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128433) (rhsResult := 128431)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 128432 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩) (none) 128431) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128434

namespace LeftMerge128435
def owner : Owner := ⟨.program ⟨257⟩, ⟨16519⟩⟩
def mergeEvent : Nat := 128435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩] } }
def rhsRaw : List Term := Proof.Events501.exact128431RawTerms
def group : MergeGroup := .relation 128433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128433) (rhsResult := 128431)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 128432 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩) (none) 128431) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128435

namespace LeftMerge128436
def owner : Owner := ⟨.program ⟨257⟩, ⟨16519⟩⟩
def mergeEvent : Nat := 128436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16965⟩⟩] } }
def rhsRaw : List Term := Proof.Events501.exact128431RawTerms
def group : MergeGroup := .relation 128433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128433) (rhsResult := 128431)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 128432 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩) (none) 128431) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16965⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128436

namespace LeftMerge128437
def owner : Owner := ⟨.program ⟨257⟩, ⟨16519⟩⟩
def mergeEvent : Nat := 128437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events501.exact128431RawTerms
def group : MergeGroup := .relation 128433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128433) (rhsResult := 128431)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 128432 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16516⟩⟩]⟩) (none) 128431) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128437

namespace LeftMerge128442
def owner : Owner := ⟨.program ⟨257⟩, ⟨17652⟩⟩
def mergeEvent : Nat := 128442
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩] } }
def leftRaw : List Term := Proof.Events501.exact128438RawTerms
def rightRaw : List Term := Proof.Events501.exact128260RawTerms
def group : MergeGroup := .operator 128438 128260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128438) (leftOrdinal := 0)
    (rightResult := 128260) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17649⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128442

namespace LeftMerge128443
def owner : Owner := ⟨.program ⟨257⟩, ⟨17652⟩⟩
def mergeEvent : Nat := 128443
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16965⟩⟩] } }
def leftRaw : List Term := Proof.Events501.exact128438RawTerms
def rightRaw : List Term := Proof.Events501.exact128260RawTerms
def group : MergeGroup := .operator 128438 128260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128438) (leftOrdinal := 2)
    (rightResult := 128260) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16965⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16965⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨15756⟩⟩], [⟨.program ⟨257⟩, ⟨16965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128443

namespace LeftMerge128536
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 17)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128536

namespace LeftMerge128537
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨48311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 29)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨48311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨48311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128537

namespace LeftMerge128539
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨48311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events467.exact119750RawTerms
def group : MergeGroup := .relation 128538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 128538) (rhsResult := 119750)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨48311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 119750) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨48311⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge128539

namespace LeftMerge128540
def owner : Owner := ⟨.program ⟨257⟩, ⟨71115⟩⟩
def mergeEvent : Nat := 128540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events502.exact128530RawTerms
def rightRaw : List Term := Proof.Events467.exact119753RawTerms
def group : MergeGroup := .operator 128530 119753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 128530) (leftOrdinal := 16)
    (rightResult := 119753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge128540

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
