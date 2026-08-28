import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge258420
def owner : Owner := ⟨.program ⟨257⟩, ⟨32342⟩⟩
def mergeEvent : Nat := 258420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32919⟩⟩] } }
def rhsRaw : List Term := Proof.Events1009.exact258415RawTerms
def group : MergeGroup := .relation 258417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258417) (rhsResult := 258415)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩]⟩) (none) 258415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨32919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258420

namespace LeftMerge258421
def owner : Owner := ⟨.program ⟨257⟩, ⟨32342⟩⟩
def mergeEvent : Nat := 258421
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1009.exact258415RawTerms
def group : MergeGroup := .relation 258417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258417) (rhsResult := 258415)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258416 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32339⟩⟩]⟩) (none) 258415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258421

namespace LeftMerge258426
def owner : Owner := ⟨.program ⟨257⟩, ⟨33406⟩⟩
def mergeEvent : Nat := 258426
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32919⟩⟩] } }
def leftRaw : List Term := Proof.Events1009.exact258422RawTerms
def rightRaw : List Term := Proof.Events1008.exact258236RawTerms
def group : MergeGroup := .operator 258422 258236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258422) (leftOrdinal := 2)
    (rightResult := 258236) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32919⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24230⟩⟩, ⟨.program ⟨257⟩, ⟨31350⟩⟩], [⟨.program ⟨257⟩, ⟨32919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258426

namespace LeftMerge258427
def owner : Owner := ⟨.program ⟨257⟩, ⟨33406⟩⟩
def mergeEvent : Nat := 258427
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩] } }
def leftRaw : List Term := Proof.Events1009.exact258422RawTerms
def rightRaw : List Term := Proof.Events1008.exact258236RawTerms
def group : MergeGroup := .operator 258422 258236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258422) (leftOrdinal := 1)
    (rightResult := 258236) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33404⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258427

namespace LeftMerge258435
def owner : Owner := ⟨.program ⟨257⟩, ⟨33739⟩⟩
def mergeEvent : Nat := 258435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩] } }
def leftRaw : List Term := Proof.Events1009.exact258429RawTerms
def rightRaw : List Term := Proof.Events1008.exact258152RawTerms
def group : MergeGroup := .operator 258429 258152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258429) (leftOrdinal := 0)
    (rightResult := 258152) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33737⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258435

namespace LeftMerge258436
def owner : Owner := ⟨.program ⟨257⟩, ⟨33739⟩⟩
def mergeEvent : Nat := 258436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩] } }
def leftRaw : List Term := Proof.Events1009.exact258429RawTerms
def rightRaw : List Term := Proof.Events1008.exact258152RawTerms
def group : MergeGroup := .operator 258429 258152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258429) (leftOrdinal := 1)
    (rightResult := 258152) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33737⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258436

namespace LeftMerge258438
def owner : Owner := ⟨.program ⟨257⟩, ⟨33739⟩⟩
def mergeEvent : Nat := 258438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33056⟩⟩] } }
def rhsRaw : List Term := Proof.Events1008.exact258149RawTerms
def group : MergeGroup := .relation 258437
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258437) (rhsResult := 258149)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33737⟩⟩) ⟨33056⟩ 258149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33056⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨33056⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258438

namespace LeftMerge258452
def owner : Owner := ⟨.program ⟨257⟩, ⟨32599⟩⟩
def mergeEvent : Nat := 258452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32596⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1009.exact258446RawTerms
def group : MergeGroup := .operator 251495 258446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 258446) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32596⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32596⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258452

namespace LeftMerge258573
def owner : Owner := ⟨.program ⟨257⟩, ⟨33288⟩⟩
def mergeEvent : Nat := 258573
def frameStart : Nat := 258507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1010.exact258569RawTerms
def rightRaw : List Term := Proof.Events1010.exact258567RawTerms
def group : MergeGroup := .operator 258569 258567
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258569) (leftOrdinal := 0)
    (rightResult := 258567) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258573

namespace LeftMerge258585
def owner : Owner := ⟨.program ⟨257⟩, ⟨33738⟩⟩
def mergeEvent : Nat := 258585
def frameStart : Nat := 258507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩] } }
def leftRaw : List Term := Proof.Events1010.exact258581RawTerms
def rightRaw : List Term := Proof.Events1009.exact258558RawTerms
def group : MergeGroup := .operator 258581 258558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258581) (leftOrdinal := 0)
    (rightResult := 258558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33737⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258585

namespace LeftMerge258586
def owner : Owner := ⟨.program ⟨257⟩, ⟨33738⟩⟩
def mergeEvent : Nat := 258586
def frameStart : Nat := 258507
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩] } }
def leftRaw : List Term := Proof.Events1010.exact258581RawTerms
def rightRaw : List Term := Proof.Events1009.exact258558RawTerms
def group : MergeGroup := .operator 258581 258558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258581) (leftOrdinal := 1)
    (rightResult := 258558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33737⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258586

namespace LeftMerge258588
def owner : Owner := ⟨.program ⟨257⟩, ⟨33738⟩⟩
def mergeEvent : Nat := 258588
def frameStart : Nat := 258507
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33056⟩⟩] } }
def rhsRaw : List Term := Proof.Events1009.exact258555RawTerms
def group : MergeGroup := .relation 258587
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258587) (rhsResult := 258555)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33737⟩⟩) ⟨33056⟩ 258555) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33056⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨33056⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258588

namespace LeftMerge258596
def owner : Owner := ⟨.program ⟨257⟩, ⟨32013⟩⟩
def mergeEvent : Nat := 258596
def frameStart : Nat := 258507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32011⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1010.exact258569RawTerms
def rightRaw : List Term := Proof.Events1010.exact258592RawTerms
def group : MergeGroup := .operator 258569 258592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258569) (leftOrdinal := 0)
    (rightResult := 258592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32011⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32011⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258596

namespace LeftMerge258613
def owner : Owner := ⟨.program ⟨257⟩, ⟨32599⟩⟩
def mergeEvent : Nat := 258613
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }
def rhsRaw : List Term := Proof.Events1010.exact258610RawTerms
def group : MergeGroup := .relation 258612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258612) (rhsResult := 258610)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258611 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32596⟩⟩]⟩) (none) 258610) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258613

namespace LeftMerge258614
def owner : Owner := ⟨.program ⟨257⟩, ⟨32599⟩⟩
def mergeEvent : Nat := 258614
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩] } }
def rhsRaw : List Term := Proof.Events1010.exact258610RawTerms
def group : MergeGroup := .relation 258612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258612) (rhsResult := 258610)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258611 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32596⟩⟩]⟩) (none) 258610) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33737⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge258614

namespace LeftMerge258615
def owner : Owner := ⟨.program ⟨257⟩, ⟨32599⟩⟩
def mergeEvent : Nat := 258615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33056⟩⟩] } }
def rhsRaw : List Term := Proof.Events1010.exact258610RawTerms
def group : MergeGroup := .relation 258612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 258612) (rhsResult := 258610)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32596⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 258611 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32596⟩⟩]⟩) (none) 258610) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31788⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33056⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨31788⟩⟩], [⟨.program ⟨257⟩, ⟨33056⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258615

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
