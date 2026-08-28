import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge238464
def owner : Owner := ⟨.program ⟨257⟩, ⟨41600⟩⟩
def mergeEvent : Nat := 238464
def frameStart : Nat := 238371
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩] } }
def leftRaw : List Term := Proof.Events931.exact238459RawTerms
def rightRaw : List Term := Proof.Events931.exact238416RawTerms
def group : MergeGroup := .operator 238459 238416
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238459) (leftOrdinal := 1)
    (rightResult := 238416) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41597⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238464

namespace LeftMerge238466
def owner : Owner := ⟨.program ⟨257⟩, ⟨41600⟩⟩
def mergeEvent : Nat := 238466
def frameStart : Nat := 238371
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41097⟩⟩] } }
def rhsRaw : List Term := Proof.Events931.exact238413RawTerms
def group : MergeGroup := .relation 238465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238465) (rhsResult := 238413)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41597⟩⟩) ⟨41097⟩ 238413) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41097⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨41097⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238466

namespace LeftMerge238474
def owner : Owner := ⟨.program ⟨257⟩, ⟨40094⟩⟩
def mergeEvent : Nat := 238474
def frameStart : Nat := 238371
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events931.exact238427RawTerms
def rightRaw : List Term := Proof.Events931.exact238470RawTerms
def group : MergeGroup := .operator 238427 238470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238427) (leftOrdinal := 0)
    (rightResult := 238470) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238474

namespace LeftMerge238491
def owner : Owner := ⟨.program ⟨257⟩, ⟨40532⟩⟩
def mergeEvent : Nat := 238491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events931.exact238488RawTerms
def group : MergeGroup := .relation 238490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238490) (rhsResult := 238488)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238489 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩) (none) 238488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238491

namespace LeftMerge238492
def owner : Owner := ⟨.program ⟨257⟩, ⟨40532⟩⟩
def mergeEvent : Nat := 238492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩] } }
def rhsRaw : List Term := Proof.Events931.exact238488RawTerms
def group : MergeGroup := .relation 238490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238490) (rhsResult := 238488)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238489 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩) (none) 238488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238492

namespace LeftMerge238493
def owner : Owner := ⟨.program ⟨257⟩, ⟨40532⟩⟩
def mergeEvent : Nat := 238493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41097⟩⟩] } }
def rhsRaw : List Term := Proof.Events931.exact238488RawTerms
def group : MergeGroup := .relation 238490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238490) (rhsResult := 238488)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238489 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩) (none) 238488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41097⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨41097⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238493

namespace LeftMerge238494
def owner : Owner := ⟨.program ⟨257⟩, ⟨40532⟩⟩
def mergeEvent : Nat := 238494
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events931.exact238488RawTerms
def group : MergeGroup := .relation 238490
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238490) (rhsResult := 238488)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238489 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40529⟩⟩]⟩) (none) 238488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238494

namespace LeftMerge238499
def owner : Owner := ⟨.program ⟨257⟩, ⟨41599⟩⟩
def mergeEvent : Nat := 238499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41097⟩⟩] } }
def leftRaw : List Term := Proof.Events931.exact238495RawTerms
def rightRaw : List Term := Proof.Events930.exact238309RawTerms
def group : MergeGroup := .operator 238495 238309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238495) (leftOrdinal := 2)
    (rightResult := 238309) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41097⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41097⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14151⟩⟩, ⟨.program ⟨257⟩, ⟨39746⟩⟩], [⟨.program ⟨257⟩, ⟨41097⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238499

namespace LeftMerge238500
def owner : Owner := ⟨.program ⟨257⟩, ⟨41599⟩⟩
def mergeEvent : Nat := 238500
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩] } }
def leftRaw : List Term := Proof.Events931.exact238495RawTerms
def rightRaw : List Term := Proof.Events930.exact238309RawTerms
def group : MergeGroup := .operator 238495 238309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238495) (leftOrdinal := 1)
    (rightResult := 238309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41597⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238500

namespace LeftMerge238508
def owner : Owner := ⟨.program ⟨257⟩, ⟨41941⟩⟩
def mergeEvent : Nat := 238508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩] } }
def leftRaw : List Term := Proof.Events931.exact238502RawTerms
def rightRaw : List Term := Proof.Events930.exact238225RawTerms
def group : MergeGroup := .operator 238502 238225
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238502) (leftOrdinal := 0)
    (rightResult := 238225) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41939⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238508

namespace LeftMerge238509
def owner : Owner := ⟨.program ⟨257⟩, ⟨41941⟩⟩
def mergeEvent : Nat := 238509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩] } }
def leftRaw : List Term := Proof.Events931.exact238502RawTerms
def rightRaw : List Term := Proof.Events930.exact238225RawTerms
def group : MergeGroup := .operator 238502 238225
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238502) (leftOrdinal := 1)
    (rightResult := 238225) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41939⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238509

namespace LeftMerge238511
def owner : Owner := ⟨.program ⟨257⟩, ⟨41941⟩⟩
def mergeEvent : Nat := 238511
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41243⟩⟩] } }
def rhsRaw : List Term := Proof.Events930.exact238222RawTerms
def group : MergeGroup := .relation 238510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238510) (rhsResult := 238222)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41939⟩⟩) ⟨41243⟩ 238222) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41243⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨41243⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238511

namespace LeftMerge238525
def owner : Owner := ⟨.program ⟨257⟩, ⟨40819⟩⟩
def mergeEvent : Nat := 238525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40816⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events931.exact238519RawTerms
def group : MergeGroup := .operator 236870 238519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 238519) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40816⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40816⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238525

namespace LeftMerge238646
def owner : Owner := ⟨.program ⟨257⟩, ⟨41460⟩⟩
def mergeEvent : Nat := 238646
def frameStart : Nat := 238580
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events932.exact238642RawTerms
def rightRaw : List Term := Proof.Events932.exact238640RawTerms
def group : MergeGroup := .operator 238642 238640
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238642) (leftOrdinal := 0)
    (rightResult := 238640) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238646

namespace LeftMerge238658
def owner : Owner := ⟨.program ⟨257⟩, ⟨41940⟩⟩
def mergeEvent : Nat := 238658
def frameStart : Nat := 238580
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩] } }
def leftRaw : List Term := Proof.Events932.exact238654RawTerms
def rightRaw : List Term := Proof.Events932.exact238631RawTerms
def group : MergeGroup := .operator 238654 238631
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238654) (leftOrdinal := 0)
    (rightResult := 238631) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41939⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238658

namespace LeftMerge238659
def owner : Owner := ⟨.program ⟨257⟩, ⟨41940⟩⟩
def mergeEvent : Nat := 238659
def frameStart : Nat := 238580
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩] } }
def leftRaw : List Term := Proof.Events932.exact238654RawTerms
def rightRaw : List Term := Proof.Events932.exact238631RawTerms
def group : MergeGroup := .operator 238654 238631
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238654) (leftOrdinal := 1)
    (rightResult := 238631) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41939⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41939⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238659

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
