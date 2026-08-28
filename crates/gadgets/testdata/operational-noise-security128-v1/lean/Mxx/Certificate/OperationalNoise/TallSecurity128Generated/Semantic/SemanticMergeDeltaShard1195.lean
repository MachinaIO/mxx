import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge194422
def owner : Owner := ⟨.program ⟨257⟩, ⟨39849⟩⟩
def mergeEvent : Nat := 194422
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def leftRaw : List Term := Proof.Events759.exact194418RawTerms
def rightRaw : List Term := Proof.Events759.exact194388RawTerms
def group : MergeGroup := .operator 194418 194388
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194418) (leftOrdinal := 1)
    (rightResult := 194388) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194422

namespace LeftMerge194430
def owner : Owner := ⟨.program ⟨257⟩, ⟨41642⟩⟩
def mergeEvent : Nat := 194430
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩] } }
def leftRaw : List Term := Proof.Events759.exact194424RawTerms
def rightRaw : List Term := Proof.Events759.exact194360RawTerms
def group : MergeGroup := .operator 194424 194360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194424) (leftOrdinal := 1)
    (rightResult := 194360) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41641⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194430

namespace LeftMerge194432
def owner : Owner := ⟨.program ⟨257⟩, ⟨41642⟩⟩
def mergeEvent : Nat := 194432
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41121⟩⟩] } }
def rhsRaw : List Term := Proof.Events759.exact194357RawTerms
def group : MergeGroup := .relation 194431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194431) (rhsResult := 194357)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41641⟩⟩) ⟨41121⟩ 194357) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41121⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], [⟨.program ⟨257⟩, ⟨41121⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194432

namespace LeftMerge194433
def owner : Owner := ⟨.program ⟨257⟩, ⟨41642⟩⟩
def mergeEvent : Nat := 194433
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩] } }
def leftRaw : List Term := Proof.Events759.exact194424RawTerms
def rightRaw : List Term := Proof.Events759.exact194360RawTerms
def group : MergeGroup := .operator 194424 194360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194424) (leftOrdinal := 0)
    (rightResult := 194360) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41641⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194433

namespace LeftMerge194447
def owner : Owner := ⟨.program ⟨257⟩, ⟨40572⟩⟩
def mergeEvent : Nat := 194447
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events759.exact194441RawTerms
def group : MergeGroup := .operator 192995 194441
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 194441) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40569⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194447

namespace LeftMerge194526
def owner : Owner := ⟨.program ⟨257⟩, ⟨39843⟩⟩
def mergeEvent : Nat := 194526
def frameStart : Nat := 194496
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events759.exact194522RawTerms
def rightRaw : List Term := Proof.Events759.exact194519RawTerms
def group : MergeGroup := .operator 194522 194519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194522) (leftOrdinal := 0)
    (rightResult := 194519) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14211⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194526

namespace LeftMerge194556
def owner : Owner := ⟨.program ⟨257⟩, ⟨41396⟩⟩
def mergeEvent : Nat := 194556
def frameStart : Nat := 194496
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events759.exact194552RawTerms
def rightRaw : List Term := Proof.Events759.exact194550RawTerms
def group : MergeGroup := .operator 194552 194550
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194552) (leftOrdinal := 0)
    (rightResult := 194550) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194556

namespace LeftMerge194579
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 194579
def frameStart : Nat := 194496
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events760.exact194575RawTerms
def rightRaw : List Term := Proof.Events760.exact194572RawTerms
def group : MergeGroup := .operator 194575 194572
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194575) (leftOrdinal := 0)
    (rightResult := 194572) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194579

namespace LeftMerge194588
def owner : Owner := ⟨.program ⟨257⟩, ⟨41644⟩⟩
def mergeEvent : Nat := 194588
def frameStart : Nat := 194496
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩] } }
def leftRaw : List Term := Proof.Events760.exact194584RawTerms
def rightRaw : List Term := Proof.Events759.exact194541RawTerms
def group : MergeGroup := .operator 194584 194541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194584) (leftOrdinal := 0)
    (rightResult := 194541) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41641⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194588

namespace LeftMerge194589
def owner : Owner := ⟨.program ⟨257⟩, ⟨41644⟩⟩
def mergeEvent : Nat := 194589
def frameStart : Nat := 194496
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩] } }
def leftRaw : List Term := Proof.Events760.exact194584RawTerms
def rightRaw : List Term := Proof.Events759.exact194541RawTerms
def group : MergeGroup := .operator 194584 194541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194584) (leftOrdinal := 1)
    (rightResult := 194541) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41641⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194589

namespace LeftMerge194591
def owner : Owner := ⟨.program ⟨257⟩, ⟨41644⟩⟩
def mergeEvent : Nat := 194591
def frameStart : Nat := 194496
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41121⟩⟩] } }
def rhsRaw : List Term := Proof.Events759.exact194538RawTerms
def group : MergeGroup := .relation 194590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194590) (rhsResult := 194538)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41641⟩⟩) ⟨41121⟩ 194538) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41121⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], [⟨.program ⟨257⟩, ⟨41121⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194591

namespace LeftMerge194599
def owner : Owner := ⟨.program ⟨257⟩, ⟨40126⟩⟩
def mergeEvent : Nat := 194599
def frameStart : Nat := 194496
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events759.exact194552RawTerms
def rightRaw : List Term := Proof.Events760.exact194595RawTerms
def group : MergeGroup := .operator 194552 194595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 194552) (leftOrdinal := 0)
    (rightResult := 194595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194599

namespace LeftMerge194616
def owner : Owner := ⟨.program ⟨257⟩, ⟨40572⟩⟩
def mergeEvent : Nat := 194616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events760.exact194613RawTerms
def group : MergeGroup := .relation 194615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194615) (rhsResult := 194613)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 194614 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩]⟩) (none) 194613) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194616

namespace LeftMerge194617
def owner : Owner := ⟨.program ⟨257⟩, ⟨40572⟩⟩
def mergeEvent : Nat := 194617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩] } }
def rhsRaw : List Term := Proof.Events760.exact194613RawTerms
def group : MergeGroup := .relation 194615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194615) (rhsResult := 194613)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 194614 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩]⟩) (none) 194613) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41641⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194617

namespace LeftMerge194618
def owner : Owner := ⟨.program ⟨257⟩, ⟨40572⟩⟩
def mergeEvent : Nat := 194618
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41121⟩⟩] } }
def rhsRaw : List Term := Proof.Events760.exact194613RawTerms
def group : MergeGroup := .relation 194615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194615) (rhsResult := 194613)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 194614 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩]⟩) (none) 194613) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41121⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14211⟩⟩, ⟨.program ⟨257⟩, ⟨39842⟩⟩], [⟨.program ⟨257⟩, ⟨41121⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge194618

namespace LeftMerge194619
def owner : Owner := ⟨.program ⟨257⟩, ⟨40572⟩⟩
def mergeEvent : Nat := 194619
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events760.exact194613RawTerms
def group : MergeGroup := .relation 194615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 194615) (rhsResult := 194613)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 194614 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40569⟩⟩]⟩) (none) 194613) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40124⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨40124⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge194619

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
