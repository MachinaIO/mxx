import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge95525
def owner : Owner := ⟨.program ⟨257⟩, ⟨59621⟩⟩
def mergeEvent : Nat := 95525
def frameStart : Nat := 95495
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events373.exact95521RawTerms
def rightRaw : List Term := Proof.Events373.exact95518RawTerms
def group : MergeGroup := .operator 95521 95518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95521) (leftOrdinal := 0)
    (rightResult := 95518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25310⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95525

namespace LeftMerge95555
def owner : Owner := ⟨.program ⟨257⟩, ⟨61248⟩⟩
def mergeEvent : Nat := 95555
def frameStart : Nat := 95495
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95551RawTerms
def rightRaw : List Term := Proof.Events373.exact95549RawTerms
def group : MergeGroup := .operator 95551 95549
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95551) (leftOrdinal := 0)
    (rightResult := 95549) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95555

namespace LeftMerge95578
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 95578
def frameStart : Nat := 95495
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95574RawTerms
def rightRaw : List Term := Proof.Events373.exact95571RawTerms
def group : MergeGroup := .operator 95574 95571
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95574) (leftOrdinal := 0)
    (rightResult := 95571) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95578

namespace LeftMerge95587
def owner : Owner := ⟨.program ⟨257⟩, ⟨61517⟩⟩
def mergeEvent : Nat := 95587
def frameStart : Nat := 95495
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95583RawTerms
def rightRaw : List Term := Proof.Events373.exact95540RawTerms
def group : MergeGroup := .operator 95583 95540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95583) (leftOrdinal := 0)
    (rightResult := 95540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61514⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95587

namespace LeftMerge95588
def owner : Owner := ⟨.program ⟨257⟩, ⟨61517⟩⟩
def mergeEvent : Nat := 95588
def frameStart : Nat := 95495
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95583RawTerms
def rightRaw : List Term := Proof.Events373.exact95540RawTerms
def group : MergeGroup := .operator 95583 95540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95583) (leftOrdinal := 1)
    (rightResult := 95540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61514⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95588

namespace LeftMerge95590
def owner : Owner := ⟨.program ⟨257⟩, ⟨61517⟩⟩
def mergeEvent : Nat := 95590
def frameStart : Nat := 95495
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60979⟩⟩] } }
def rhsRaw : List Term := Proof.Events373.exact95537RawTerms
def group : MergeGroup := .relation 95589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95589) (rhsResult := 95537)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61514⟩⟩) ⟨60979⟩ 95537) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60979⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], [⟨.program ⟨257⟩, ⟨60979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95590

namespace LeftMerge95598
def owner : Owner := ⟨.program ⟨257⟩, ⟨59870⟩⟩
def mergeEvent : Nat := 95598
def frameStart : Nat := 95495
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95551RawTerms
def rightRaw : List Term := Proof.Events373.exact95594RawTerms
def group : MergeGroup := .operator 95551 95594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95551) (leftOrdinal := 0)
    (rightResult := 95594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95598

namespace LeftMerge95615
def owner : Owner := ⟨.program ⟨257⟩, ⟨60442⟩⟩
def mergeEvent : Nat := 95615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events373.exact95612RawTerms
def group : MergeGroup := .relation 95614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95614) (rhsResult := 95612)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95613 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60439⟩⟩]⟩) (none) 95612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95615

namespace LeftMerge95616
def owner : Owner := ⟨.program ⟨257⟩, ⟨60442⟩⟩
def mergeEvent : Nat := 95616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩] } }
def rhsRaw : List Term := Proof.Events373.exact95612RawTerms
def group : MergeGroup := .relation 95614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95614) (rhsResult := 95612)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95613 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60439⟩⟩]⟩) (none) 95612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95616

namespace LeftMerge95617
def owner : Owner := ⟨.program ⟨257⟩, ⟨60442⟩⟩
def mergeEvent : Nat := 95617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60979⟩⟩] } }
def rhsRaw : List Term := Proof.Events373.exact95612RawTerms
def group : MergeGroup := .relation 95614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95614) (rhsResult := 95612)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95613 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60439⟩⟩]⟩) (none) 95612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60979⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], [⟨.program ⟨257⟩, ⟨60979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95617

namespace LeftMerge95618
def owner : Owner := ⟨.program ⟨257⟩, ⟨60442⟩⟩
def mergeEvent : Nat := 95618
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events373.exact95612RawTerms
def group : MergeGroup := .relation 95614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95614) (rhsResult := 95612)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95613 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60439⟩⟩]⟩) (none) 95612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95618

namespace LeftMerge95623
def owner : Owner := ⟨.program ⟨257⟩, ⟨61516⟩⟩
def mergeEvent : Nat := 95623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60979⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95619RawTerms
def rightRaw : List Term := Proof.Events372.exact95433RawTerms
def group : MergeGroup := .operator 95619 95433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95619) (leftOrdinal := 2)
    (rightResult := 95433) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60979⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25310⟩⟩, ⟨.program ⟨257⟩, ⟨59620⟩⟩], [⟨.program ⟨257⟩, ⟨60979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95623

namespace LeftMerge95624
def owner : Owner := ⟨.program ⟨257⟩, ⟨61516⟩⟩
def mergeEvent : Nat := 95624
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95619RawTerms
def rightRaw : List Term := Proof.Events372.exact95433RawTerms
def group : MergeGroup := .operator 95619 95433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95619) (leftOrdinal := 1)
    (rightResult := 95433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61514⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95624

namespace LeftMerge95632
def owner : Owner := ⟨.program ⟨257⟩, ⟨62049⟩⟩
def mergeEvent : Nat := 95632
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62047⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95626RawTerms
def rightRaw : List Term := Proof.Events372.exact95349RawTerms
def group : MergeGroup := .operator 95626 95349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95626) (leftOrdinal := 0)
    (rightResult := 95349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62047⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨62047⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95632

namespace LeftMerge95633
def owner : Owner := ⟨.program ⟨257⟩, ⟨62049⟩⟩
def mergeEvent : Nat := 95633
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62047⟩⟩] } }
def leftRaw : List Term := Proof.Events373.exact95626RawTerms
def rightRaw : List Term := Proof.Events372.exact95349RawTerms
def group : MergeGroup := .operator 95626 95349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95626) (leftOrdinal := 1)
    (rightResult := 95349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨62047⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62047⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95633

namespace LeftMerge95635
def owner : Owner := ⟨.program ⟨257⟩, ⟨62049⟩⟩
def mergeEvent : Nat := 95635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61146⟩⟩] } }
def rhsRaw : List Term := Proof.Events372.exact95346RawTerms
def group : MergeGroup := .relation 95634
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95634) (rhsResult := 95346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨62047⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨62047⟩⟩) ⟨61146⟩ 95346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61146⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨59868⟩⟩], [⟨.program ⟨257⟩, ⟨61146⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95635

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
