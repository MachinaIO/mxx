import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge120437
def owner : Owner := ⟨.program ⟨257⟩, ⟨45059⟩⟩
def mergeEvent : Nat := 120437
def frameStart : Nat := 120407
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events470.exact120433RawTerms
def rightRaw : List Term := Proof.Events470.exact120430RawTerms
def group : MergeGroup := .operator 120433 120430
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120433) (leftOrdinal := 0)
    (rightResult := 120430) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14721⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120437

namespace LeftMerge120467
def owner : Owner := ⟨.program ⟨257⟩, ⟨46732⟩⟩
def mergeEvent : Nat := 120467
def frameStart : Nat := 120407
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events470.exact120463RawTerms
def rightRaw : List Term := Proof.Events470.exact120461RawTerms
def group : MergeGroup := .operator 120463 120461
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120463) (leftOrdinal := 0)
    (rightResult := 120461) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120467

namespace LeftMerge120490
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 120490
def frameStart : Nat := 120407
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events470.exact120486RawTerms
def rightRaw : List Term := Proof.Events470.exact120483RawTerms
def group : MergeGroup := .operator 120486 120483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120486) (leftOrdinal := 0)
    (rightResult := 120483) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120490

namespace LeftMerge120499
def owner : Owner := ⟨.program ⟨257⟩, ⟨46938⟩⟩
def mergeEvent : Nat := 120499
def frameStart : Nat := 120407
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩] } }
def leftRaw : List Term := Proof.Events470.exact120495RawTerms
def rightRaw : List Term := Proof.Events470.exact120452RawTerms
def group : MergeGroup := .operator 120495 120452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120495) (leftOrdinal := 0)
    (rightResult := 120452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46935⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120499

namespace LeftMerge120500
def owner : Owner := ⟨.program ⟨257⟩, ⟨46938⟩⟩
def mergeEvent : Nat := 120500
def frameStart : Nat := 120407
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩] } }
def leftRaw : List Term := Proof.Events470.exact120495RawTerms
def rightRaw : List Term := Proof.Events470.exact120452RawTerms
def group : MergeGroup := .operator 120495 120452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120495) (leftOrdinal := 1)
    (rightResult := 120452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46935⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120500

namespace LeftMerge120502
def owner : Owner := ⟨.program ⟨257⟩, ⟨46938⟩⟩
def mergeEvent : Nat := 120502
def frameStart : Nat := 120407
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46445⟩⟩] } }
def rhsRaw : List Term := Proof.Events470.exact120449RawTerms
def group : MergeGroup := .relation 120501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120501) (rhsResult := 120449)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46935⟩⟩) ⟨46445⟩ 120449) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46445⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], [⟨.program ⟨257⟩, ⟨46445⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120502

namespace LeftMerge120510
def owner : Owner := ⟨.program ⟨257⟩, ⟨45438⟩⟩
def mergeEvent : Nat := 120510
def frameStart : Nat := 120407
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events470.exact120463RawTerms
def rightRaw : List Term := Proof.Events470.exact120506RawTerms
def group : MergeGroup := .operator 120463 120506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120463) (leftOrdinal := 0)
    (rightResult := 120506) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120510

namespace LeftMerge120527
def owner : Owner := ⟨.program ⟨257⟩, ⟨45872⟩⟩
def mergeEvent : Nat := 120527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events470.exact120524RawTerms
def group : MergeGroup := .relation 120526
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120526) (rhsResult := 120524)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120525 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45869⟩⟩]⟩) (none) 120524) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120527

namespace LeftMerge120528
def owner : Owner := ⟨.program ⟨257⟩, ⟨45872⟩⟩
def mergeEvent : Nat := 120528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩] } }
def rhsRaw : List Term := Proof.Events470.exact120524RawTerms
def group : MergeGroup := .relation 120526
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120526) (rhsResult := 120524)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120525 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45869⟩⟩]⟩) (none) 120524) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120528

namespace LeftMerge120529
def owner : Owner := ⟨.program ⟨257⟩, ⟨45872⟩⟩
def mergeEvent : Nat := 120529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46445⟩⟩] } }
def rhsRaw : List Term := Proof.Events470.exact120524RawTerms
def group : MergeGroup := .relation 120526
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120526) (rhsResult := 120524)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120525 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45869⟩⟩]⟩) (none) 120524) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46445⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], [⟨.program ⟨257⟩, ⟨46445⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120529

namespace LeftMerge120530
def owner : Owner := ⟨.program ⟨257⟩, ⟨45872⟩⟩
def mergeEvent : Nat := 120530
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events470.exact120524RawTerms
def group : MergeGroup := .relation 120526
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120526) (rhsResult := 120524)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120525 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45869⟩⟩]⟩) (none) 120524) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120530

namespace LeftMerge120535
def owner : Owner := ⟨.program ⟨257⟩, ⟨46937⟩⟩
def mergeEvent : Nat := 120535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46445⟩⟩] } }
def leftRaw : List Term := Proof.Events470.exact120531RawTerms
def rightRaw : List Term := Proof.Events470.exact120345RawTerms
def group : MergeGroup := .operator 120531 120345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120531) (leftOrdinal := 2)
    (rightResult := 120345) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46445⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46445⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14721⟩⟩, ⟨.program ⟨257⟩, ⟨45058⟩⟩], [⟨.program ⟨257⟩, ⟨46445⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120535

namespace LeftMerge120536
def owner : Owner := ⟨.program ⟨257⟩, ⟨46937⟩⟩
def mergeEvent : Nat := 120536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩] } }
def leftRaw : List Term := Proof.Events470.exact120531RawTerms
def rightRaw : List Term := Proof.Events470.exact120345RawTerms
def group : MergeGroup := .operator 120531 120345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120531) (leftOrdinal := 1)
    (rightResult := 120345) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46935⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120536

namespace LeftMerge120544
def owner : Owner := ⟨.program ⟨257⟩, ⟨47251⟩⟩
def mergeEvent : Nat := 120544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩] } }
def leftRaw : List Term := Proof.Events470.exact120538RawTerms
def rightRaw : List Term := Proof.Events469.exact120261RawTerms
def group : MergeGroup := .operator 120538 120261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120538) (leftOrdinal := 0)
    (rightResult := 120261) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120544

namespace LeftMerge120545
def owner : Owner := ⟨.program ⟨257⟩, ⟨47251⟩⟩
def mergeEvent : Nat := 120545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩] } }
def leftRaw : List Term := Proof.Events470.exact120538RawTerms
def rightRaw : List Term := Proof.Events469.exact120261RawTerms
def group : MergeGroup := .operator 120538 120261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120538) (leftOrdinal := 1)
    (rightResult := 120261) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120545

namespace LeftMerge120547
def owner : Owner := ⟨.program ⟨257⟩, ⟨47251⟩⟩
def mergeEvent : Nat := 120547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46585⟩⟩] } }
def rhsRaw : List Term := Proof.Events469.exact120258RawTerms
def group : MergeGroup := .relation 120546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120546) (rhsResult := 120258)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47249⟩⟩) ⟨46585⟩ 120258) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46585⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨46585⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120547

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
