import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge251435
def owner : Owner := ⟨.program ⟨257⟩, ⟨15007⟩⟩
def mergeEvent : Nat := 251435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events047.exact12065RawTerms
def rightRaw : List Term := Proof.Events982.exact251403RawTerms
def group : MergeGroup := .operator 12065 251403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12065) (leftOrdinal := 0)
    (rightResult := 251403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251435

namespace LeftMerge251440
def owner : Owner := ⟨.program ⟨257⟩, ⟨8038⟩⟩
def mergeEvent : Nat := 251440
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251273RawTerms
def rightRaw : List Term := Proof.Events066.exact17106RawTerms
def group : MergeGroup := .operator 251273 17106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251273) (leftOrdinal := 0)
    (rightResult := 17106) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251440

namespace LeftMerge251457
def owner : Owner := ⟨.program ⟨257⟩, ⟨15010⟩⟩
def mergeEvent : Nat := 251457
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251451RawTerms
def rightRaw : List Term := Proof.Events066.exact17095RawTerms
def group : MergeGroup := .operator 251451 17095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251451) (leftOrdinal := 1)
    (rightResult := 17095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251457

namespace LeftMerge251459
def owner : Owner := ⟨.program ⟨257⟩, ⟨15010⟩⟩
def mergeEvent : Nat := 251459
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def rhsRaw : List Term := Proof.Events066.exact17065RawTerms
def group : MergeGroup := .relation 251458
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251458) (rhsResult := 17065)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9565⟩⟩) ⟨7285⟩ 17065) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251459

namespace LeftMerge251460
def owner : Owner := ⟨.program ⟨257⟩, ⟨15010⟩⟩
def mergeEvent : Nat := 251460
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251451RawTerms
def rightRaw : List Term := Proof.Events066.exact17095RawTerms
def group : MergeGroup := .operator 251451 17095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251451) (leftOrdinal := 0)
    (rightResult := 17095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251460

namespace LeftMerge251465
def owner : Owner := ⟨.program ⟨257⟩, ⟨47721⟩⟩
def mergeEvent : Nat := 251465
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251461RawTerms
def rightRaw : List Term := Proof.Events982.exact251431RawTerms
def group : MergeGroup := .operator 251461 251431
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251461) (leftOrdinal := 1)
    (rightResult := 251431) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251465

namespace LeftMerge251473
def owner : Owner := ⟨.program ⟨257⟩, ⟨49605⟩⟩
def mergeEvent : Nat := 251473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251467RawTerms
def rightRaw : List Term := Proof.Events982.exact251398RawTerms
def group : MergeGroup := .operator 251467 251398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251467) (leftOrdinal := 1)
    (rightResult := 251398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49604⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251473

namespace LeftMerge251475
def owner : Owner := ⟨.program ⟨257⟩, ⟨49605⟩⟩
def mergeEvent : Nat := 251475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49119⟩⟩] } }
def rhsRaw : List Term := Proof.Events982.exact251395RawTerms
def group : MergeGroup := .relation 251474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 251474) (rhsResult := 251395)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49604⟩⟩) ⟨49119⟩ 251395) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49119⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨49119⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251475

namespace LeftMerge251476
def owner : Owner := ⟨.program ⟨257⟩, ⟨49605⟩⟩
def mergeEvent : Nat := 251476
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251467RawTerms
def rightRaw : List Term := Proof.Events982.exact251398RawTerms
def group : MergeGroup := .operator 251467 251398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251467) (leftOrdinal := 0)
    (rightResult := 251398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49604⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251476

namespace LeftMerge251488
def owner : Owner := ⟨.program ⟨257⟩, ⟨5508⟩⟩
def mergeEvent : Nat := 251488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251273RawTerms
def rightRaw : List Term := Proof.Events067.exact17158RawTerms
def group : MergeGroup := .operator 251273 17158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251273) (leftOrdinal := 0)
    (rightResult := 17158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251488

namespace LeftMerge251501
def owner : Owner := ⟨.program ⟨257⟩, ⟨48542⟩⟩
def mergeEvent : Nat := 251501
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events982.exact251484RawTerms
def group : MergeGroup := .operator 251495 251484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 251484) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48539⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48539⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251501

namespace LeftMerge251580
def owner : Owner := ⟨.program ⟨257⟩, ⟨47715⟩⟩
def mergeEvent : Nat := 251580
def frameStart : Nat := 251550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events982.exact251576RawTerms
def rightRaw : List Term := Proof.Events982.exact251573RawTerms
def group : MergeGroup := .operator 251576 251573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251576) (leftOrdinal := 0)
    (rightResult := 251573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251580

namespace LeftMerge251610
def owner : Owner := ⟨.program ⟨257⟩, ⟨49408⟩⟩
def mergeEvent : Nat := 251610
def frameStart : Nat := 251550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251606RawTerms
def rightRaw : List Term := Proof.Events982.exact251604RawTerms
def group : MergeGroup := .operator 251606 251604
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251606) (leftOrdinal := 0)
    (rightResult := 251604) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251610

namespace LeftMerge251633
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 251633
def frameStart : Nat := 251550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251629RawTerms
def rightRaw : List Term := Proof.Events982.exact251626RawTerms
def group : MergeGroup := .operator 251629 251626
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251629) (leftOrdinal := 0)
    (rightResult := 251626) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251633

namespace LeftMerge251642
def owner : Owner := ⟨.program ⟨257⟩, ⟨49607⟩⟩
def mergeEvent : Nat := 251642
def frameStart : Nat := 251550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251638RawTerms
def rightRaw : List Term := Proof.Events982.exact251595RawTerms
def group : MergeGroup := .operator 251638 251595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251638) (leftOrdinal := 0)
    (rightResult := 251595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49604⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge251642

namespace LeftMerge251643
def owner : Owner := ⟨.program ⟨257⟩, ⟨49607⟩⟩
def mergeEvent : Nat := 251643
def frameStart : Nat := 251550
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251638RawTerms
def rightRaw : List Term := Proof.Events982.exact251595RawTerms
def group : MergeGroup := .operator 251638 251595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251638) (leftOrdinal := 1)
    (rightResult := 251595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49604⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15006⟩⟩, ⟨.program ⟨257⟩, ⟨47714⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49604⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge251643

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
