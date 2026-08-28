import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge212616
def owner : Owner := ⟨.program ⟨257⟩, ⟨60392⟩⟩
def mergeEvent : Nat := 212616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61459⟩⟩] } }
def rhsRaw : List Term := Proof.Events830.exact212612RawTerms
def group : MergeGroup := .relation 212614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 212614) (rhsResult := 212612)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 212613 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60389⟩⟩]⟩) (none) 212612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61459⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61459⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge212616

namespace LeftMerge212617
def owner : Owner := ⟨.program ⟨257⟩, ⟨60392⟩⟩
def mergeEvent : Nat := 212617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25250⟩⟩, ⟨.program ⟨257⟩, ⟨59485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60949⟩⟩] } }
def rhsRaw : List Term := Proof.Events830.exact212612RawTerms
def group : MergeGroup := .relation 212614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 212614) (rhsResult := 212612)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 212613 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60389⟩⟩]⟩) (none) 212612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25250⟩⟩, ⟨.program ⟨257⟩, ⟨59485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60949⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25250⟩⟩, ⟨.program ⟨257⟩, ⟨59485⟩⟩], [⟨.program ⟨257⟩, ⟨60949⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212617

namespace LeftMerge212618
def owner : Owner := ⟨.program ⟨257⟩, ⟨60392⟩⟩
def mergeEvent : Nat := 212618
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events830.exact212612RawTerms
def group : MergeGroup := .relation 212614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 212614) (rhsResult := 212612)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60389⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 212613 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60389⟩⟩]⟩) (none) 212612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge212618

namespace LeftMerge212623
def owner : Owner := ⟨.program ⟨257⟩, ⟨61461⟩⟩
def mergeEvent : Nat := 212623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25250⟩⟩, ⟨.program ⟨257⟩, ⟨59485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60949⟩⟩] } }
def leftRaw : List Term := Proof.Events830.exact212619RawTerms
def rightRaw : List Term := Proof.Events829.exact212433RawTerms
def group : MergeGroup := .operator 212619 212433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 212619) (leftOrdinal := 2)
    (rightResult := 212433) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25250⟩⟩, ⟨.program ⟨257⟩, ⟨59485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60949⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25250⟩⟩, ⟨.program ⟨257⟩, ⟨59485⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60949⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨25250⟩⟩, ⟨.program ⟨257⟩, ⟨59485⟩⟩], [⟨.program ⟨257⟩, ⟨60949⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge212623

namespace LeftMerge212624
def owner : Owner := ⟨.program ⟨257⟩, ⟨61461⟩⟩
def mergeEvent : Nat := 212624
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61459⟩⟩] } }
def leftRaw : List Term := Proof.Events830.exact212619RawTerms
def rightRaw : List Term := Proof.Events829.exact212433RawTerms
def group : MergeGroup := .operator 212619 212433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 212619) (leftOrdinal := 1)
    (rightResult := 212433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61459⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212624

namespace LeftMerge212632
def owner : Owner := ⟨.program ⟨257⟩, ⟨61894⟩⟩
def mergeEvent : Nat := 212632
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩] } }
def leftRaw : List Term := Proof.Events830.exact212626RawTerms
def rightRaw : List Term := Proof.Events829.exact212349RawTerms
def group : MergeGroup := .operator 212626 212349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 212626) (leftOrdinal := 0)
    (rightResult := 212349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212632

namespace LeftMerge212633
def owner : Owner := ⟨.program ⟨257⟩, ⟨61894⟩⟩
def mergeEvent : Nat := 212633
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩] } }
def leftRaw : List Term := Proof.Events830.exact212626RawTerms
def rightRaw : List Term := Proof.Events829.exact212349RawTerms
def group : MergeGroup := .operator 212626 212349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 212626) (leftOrdinal := 1)
    (rightResult := 212349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge212633

namespace LeftMerge212635
def owner : Owner := ⟨.program ⟨257⟩, ⟨61894⟩⟩
def mergeEvent : Nat := 212635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61101⟩⟩] } }
def rhsRaw : List Term := Proof.Events829.exact212346RawTerms
def group : MergeGroup := .relation 212634
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 212634) (rhsResult := 212346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61892⟩⟩) ⟨61101⟩ 212346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61101⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨61101⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge212635

namespace LeftMerge212649
def owner : Owner := ⟨.program ⟨257⟩, ⟨60699⟩⟩
def mergeEvent : Nat := 212649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60696⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events830.exact212643RawTerms
def group : MergeGroup := .operator 207620 212643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 212643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60696⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212649

namespace LeftMerge212770
def owner : Owner := ⟨.program ⟨257⟩, ⟨61308⟩⟩
def mergeEvent : Nat := 212770
def frameStart : Nat := 212704
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events831.exact212766RawTerms
def rightRaw : List Term := Proof.Events831.exact212764RawTerms
def group : MergeGroup := .operator 212766 212764
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 212766) (leftOrdinal := 0)
    (rightResult := 212764) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212770

namespace LeftMerge212782
def owner : Owner := ⟨.program ⟨257⟩, ⟨61893⟩⟩
def mergeEvent : Nat := 212782
def frameStart : Nat := 212704
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩] } }
def leftRaw : List Term := Proof.Events831.exact212778RawTerms
def rightRaw : List Term := Proof.Events831.exact212755RawTerms
def group : MergeGroup := .operator 212778 212755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 212778) (leftOrdinal := 0)
    (rightResult := 212755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61892⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212782

namespace LeftMerge212783
def owner : Owner := ⟨.program ⟨257⟩, ⟨61893⟩⟩
def mergeEvent : Nat := 212783
def frameStart : Nat := 212704
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩] } }
def leftRaw : List Term := Proof.Events831.exact212778RawTerms
def rightRaw : List Term := Proof.Events831.exact212755RawTerms
def group : MergeGroup := .operator 212778 212755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 212778) (leftOrdinal := 1)
    (rightResult := 212755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge212783

namespace LeftMerge212785
def owner : Owner := ⟨.program ⟨257⟩, ⟨61893⟩⟩
def mergeEvent : Nat := 212785
def frameStart : Nat := 212704
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61101⟩⟩] } }
def rhsRaw : List Term := Proof.Events831.exact212752RawTerms
def group : MergeGroup := .relation 212784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 212784) (rhsResult := 212752)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61892⟩⟩) ⟨61101⟩ 212752) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61101⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨61101⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge212785

namespace LeftMerge212793
def owner : Owner := ⟨.program ⟨257⟩, ⟨60103⟩⟩
def mergeEvent : Nat := 212793
def frameStart : Nat := 212704
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events831.exact212766RawTerms
def rightRaw : List Term := Proof.Events831.exact212789RawTerms
def group : MergeGroup := .operator 212766 212789
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 212766) (leftOrdinal := 0)
    (rightResult := 212789) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60101⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60101⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212793

namespace LeftMerge212810
def owner : Owner := ⟨.program ⟨257⟩, ⟨60699⟩⟩
def mergeEvent : Nat := 212810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }
def rhsRaw : List Term := Proof.Events831.exact212807RawTerms
def group : MergeGroup := .relation 212809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 212809) (rhsResult := 212807)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 212808 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60696⟩⟩]⟩) (none) 212807) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge212810

namespace LeftMerge212811
def owner : Owner := ⟨.program ⟨257⟩, ⟨60699⟩⟩
def mergeEvent : Nat := 212811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩] } }
def rhsRaw : List Term := Proof.Events831.exact212807RawTerms
def group : MergeGroup := .relation 212809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 212809) (rhsResult := 212807)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 212808 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60696⟩⟩]⟩) (none) 212807) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61892⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge212811

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
