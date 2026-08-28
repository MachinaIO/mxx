import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge306405
def owner : Owner := ⟨.program ⟨257⟩, ⟨61577⟩⟩
def mergeEvent : Nat := 306405
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩] } }
def leftRaw : List Term := Proof.Events1170.exact299697RawTerms
def rightRaw : List Term := Proof.Events1196.exact306399RawTerms
def group : MergeGroup := .operator 299697 306399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299697) (leftOrdinal := 0)
    (rightResult := 306399) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61575⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge306405

namespace LeftMerge306406
def owner : Owner := ⟨.program ⟨257⟩, ⟨61577⟩⟩
def mergeEvent : Nat := 306406
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩] } }
def leftRaw : List Term := Proof.Events1170.exact299697RawTerms
def rightRaw : List Term := Proof.Events1196.exact306399RawTerms
def group : MergeGroup := .operator 299697 306399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299697) (leftOrdinal := 1)
    (rightResult := 306399) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61575⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge306406

namespace LeftMerge306408
def owner : Owner := ⟨.program ⟨257⟩, ⟨61577⟩⟩
def mergeEvent : Nat := 306408
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61010⟩⟩] } }
def rhsRaw : List Term := Proof.Events1196.exact306396RawTerms
def group : MergeGroup := .relation 306407
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 306407) (rhsResult := 306396)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61575⟩⟩) ⟨61010⟩ 306396) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61010⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨61010⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge306408

namespace LeftMerge306422
def owner : Owner := ⟨.program ⟨257⟩, ⟨60495⟩⟩
def mergeEvent : Nat := 306422
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1196.exact306416RawTerms
def group : MergeGroup := .operator 295195 306416
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 306416) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60492⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge306422

namespace LeftMerge306519
def owner : Owner := ⟨.program ⟨257⟩, ⟨61268⟩⟩
def mergeEvent : Nat := 306519
def frameStart : Nat := 306465
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1197.exact306515RawTerms
def rightRaw : List Term := Proof.Events1197.exact306513RawTerms
def group : MergeGroup := .operator 306515 306513
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 306515) (leftOrdinal := 0)
    (rightResult := 306513) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge306519

namespace LeftMerge306531
def owner : Owner := ⟨.program ⟨257⟩, ⟨61576⟩⟩
def mergeEvent : Nat := 306531
def frameStart : Nat := 306465
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩] } }
def leftRaw : List Term := Proof.Events1197.exact306527RawTerms
def rightRaw : List Term := Proof.Events1197.exact306504RawTerms
def group : MergeGroup := .operator 306527 306504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 306527) (leftOrdinal := 0)
    (rightResult := 306504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61575⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge306531

namespace LeftMerge306532
def owner : Owner := ⟨.program ⟨257⟩, ⟨61576⟩⟩
def mergeEvent : Nat := 306532
def frameStart : Nat := 306465
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩] } }
def leftRaw : List Term := Proof.Events1197.exact306527RawTerms
def rightRaw : List Term := Proof.Events1197.exact306504RawTerms
def group : MergeGroup := .operator 306527 306504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 306527) (leftOrdinal := 1)
    (rightResult := 306504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61575⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge306532

namespace LeftMerge306534
def owner : Owner := ⟨.program ⟨257⟩, ⟨61576⟩⟩
def mergeEvent : Nat := 306534
def frameStart : Nat := 306465
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61010⟩⟩] } }
def rhsRaw : List Term := Proof.Events1197.exact306501RawTerms
def group : MergeGroup := .relation 306533
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 306533) (rhsResult := 306501)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61575⟩⟩) ⟨61010⟩ 306501) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61010⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨61010⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge306534

namespace LeftMerge306542
def owner : Owner := ⟨.program ⟨257⟩, ⟨59918⟩⟩
def mergeEvent : Nat := 306542
def frameStart : Nat := 306465
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59915⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1197.exact306515RawTerms
def rightRaw : List Term := Proof.Events1197.exact306538RawTerms
def group : MergeGroup := .operator 306515 306538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 306515) (leftOrdinal := 0)
    (rightResult := 306538) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59915⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge306542

namespace LeftMerge306559
def owner : Owner := ⟨.program ⟨257⟩, ⟨60495⟩⟩
def mergeEvent : Nat := 306559
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }
def rhsRaw : List Term := Proof.Events1197.exact306556RawTerms
def group : MergeGroup := .relation 306558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 306558) (rhsResult := 306556)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 306557 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩]⟩) (none) 306556) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge306559

namespace LeftMerge306560
def owner : Owner := ⟨.program ⟨257⟩, ⟨60495⟩⟩
def mergeEvent : Nat := 306560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩] } }
def rhsRaw : List Term := Proof.Events1197.exact306556RawTerms
def group : MergeGroup := .relation 306558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 306558) (rhsResult := 306556)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 306557 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩]⟩) (none) 306556) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge306560

namespace LeftMerge306561
def owner : Owner := ⟨.program ⟨257⟩, ⟨60495⟩⟩
def mergeEvent : Nat := 306561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61010⟩⟩] } }
def rhsRaw : List Term := Proof.Events1197.exact306556RawTerms
def group : MergeGroup := .relation 306558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 306558) (rhsResult := 306556)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 306557 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩]⟩) (none) 306556) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61010⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨61010⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge306561

namespace LeftMerge306562
def owner : Owner := ⟨.program ⟨257⟩, ⟨60495⟩⟩
def mergeEvent : Nat := 306562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59915⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1197.exact306556RawTerms
def group : MergeGroup := .relation 306558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 306558) (rhsResult := 306556)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 306557 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60492⟩⟩]⟩) (none) 306556) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59915⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge306562

namespace LeftMerge306567
def owner : Owner := ⟨.program ⟨257⟩, ⟨61578⟩⟩
def mergeEvent : Nat := 306567
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩] } }
def leftRaw : List Term := Proof.Events1197.exact306563RawTerms
def rightRaw : List Term := Proof.Events1196.exact306409RawTerms
def group : MergeGroup := .operator 306563 306409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 306563) (leftOrdinal := 0)
    (rightResult := 306409) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61575⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge306567

namespace LeftMerge306568
def owner : Owner := ⟨.program ⟨257⟩, ⟨61578⟩⟩
def mergeEvent : Nat := 306568
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61010⟩⟩] } }
def leftRaw : List Term := Proof.Events1197.exact306563RawTerms
def rightRaw : List Term := Proof.Events1196.exact306409RawTerms
def group : MergeGroup := .operator 306563 306409
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 306563) (leftOrdinal := 2)
    (rightResult := 306409) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61010⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61010⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨59748⟩⟩], [⟨.program ⟨257⟩, ⟨61010⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge306568

namespace LeftMerge306576
def owner : Owner := ⟨.program ⟨257⟩, ⟨61579⟩⟩
def mergeEvent : Nat := 306576
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }
def leftRaw : List Term := Proof.Events1197.exact306570RawTerms
def rightRaw : List Term := Proof.Events061.exact15742RawTerms
def group : MergeGroup := .operator 306570 15742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 306570) (leftOrdinal := 0)
    (rightResult := 15742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge306576

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
