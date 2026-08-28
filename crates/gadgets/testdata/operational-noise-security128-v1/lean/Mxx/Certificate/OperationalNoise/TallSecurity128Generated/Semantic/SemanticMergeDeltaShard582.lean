import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge97544
def owner : Owner := ⟨.program ⟨257⟩, ⟨32442⟩⟩
def mergeEvent : Nat := 97544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩] } }
def rhsRaw : List Term := Proof.Events381.exact97540RawTerms
def group : MergeGroup := .relation 97542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97542) (rhsResult := 97540)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩) (none) 97540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97544

namespace LeftMerge97545
def owner : Owner := ⟨.program ⟨257⟩, ⟨32442⟩⟩
def mergeEvent : Nat := 97545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32979⟩⟩] } }
def rhsRaw : List Term := Proof.Events381.exact97540RawTerms
def group : MergeGroup := .relation 97542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97542) (rhsResult := 97540)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩) (none) 97540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32979⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨32979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97545

namespace LeftMerge97546
def owner : Owner := ⟨.program ⟨257⟩, ⟨32442⟩⟩
def mergeEvent : Nat := 97546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events381.exact97540RawTerms
def group : MergeGroup := .relation 97542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97542) (rhsResult := 97540)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97541 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32439⟩⟩]⟩) (none) 97540) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97546

namespace LeftMerge97551
def owner : Owner := ⟨.program ⟨257⟩, ⟨33516⟩⟩
def mergeEvent : Nat := 97551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32979⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97547RawTerms
def rightRaw : List Term := Proof.Events380.exact97361RawTerms
def group : MergeGroup := .operator 97547 97361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97547) (leftOrdinal := 2)
    (rightResult := 97361) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32979⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24350⟩⟩, ⟨.program ⟨257⟩, ⟨31620⟩⟩], [⟨.program ⟨257⟩, ⟨32979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97551

namespace LeftMerge97552
def owner : Owner := ⟨.program ⟨257⟩, ⟨33516⟩⟩
def mergeEvent : Nat := 97552
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97547RawTerms
def rightRaw : List Term := Proof.Events380.exact97361RawTerms
def group : MergeGroup := .operator 97547 97361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97547) (leftOrdinal := 1)
    (rightResult := 97361) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33514⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97552

namespace LeftMerge97560
def owner : Owner := ⟨.program ⟨257⟩, ⟨34049⟩⟩
def mergeEvent : Nat := 97560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97554RawTerms
def rightRaw : List Term := Proof.Events379.exact97277RawTerms
def group : MergeGroup := .operator 97554 97277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97554) (leftOrdinal := 0)
    (rightResult := 97277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34047⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97560

namespace LeftMerge97561
def owner : Owner := ⟨.program ⟨257⟩, ⟨34049⟩⟩
def mergeEvent : Nat := 97561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97554RawTerms
def rightRaw : List Term := Proof.Events379.exact97277RawTerms
def group : MergeGroup := .operator 97554 97277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97554) (leftOrdinal := 1)
    (rightResult := 97277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34047⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97561

namespace LeftMerge97563
def owner : Owner := ⟨.program ⟨257⟩, ⟨34049⟩⟩
def mergeEvent : Nat := 97563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33146⟩⟩] } }
def rhsRaw : List Term := Proof.Events379.exact97274RawTerms
def group : MergeGroup := .relation 97562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97562) (rhsResult := 97274)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34047⟩⟩) ⟨33146⟩ 97274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33146⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨33146⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97563

namespace LeftMerge97577
def owner : Owner := ⟨.program ⟨257⟩, ⟨32799⟩⟩
def mergeEvent : Nat := 97577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32796⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events381.exact97571RawTerms
def group : MergeGroup := .operator 90620 97571
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 97571) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32796⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32796⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97577

namespace LeftMerge97698
def owner : Owner := ⟨.program ⟨257⟩, ⟨33328⟩⟩
def mergeEvent : Nat := 97698
def frameStart : Nat := 97632
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97694RawTerms
def rightRaw : List Term := Proof.Events381.exact97692RawTerms
def group : MergeGroup := .operator 97694 97692
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97694) (leftOrdinal := 0)
    (rightResult := 97692) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97698

namespace LeftMerge97710
def owner : Owner := ⟨.program ⟨257⟩, ⟨34048⟩⟩
def mergeEvent : Nat := 97710
def frameStart : Nat := 97632
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97706RawTerms
def rightRaw : List Term := Proof.Events381.exact97683RawTerms
def group : MergeGroup := .operator 97706 97683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97706) (leftOrdinal := 0)
    (rightResult := 97683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34047⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97710

namespace LeftMerge97711
def owner : Owner := ⟨.program ⟨257⟩, ⟨34048⟩⟩
def mergeEvent : Nat := 97711
def frameStart : Nat := 97632
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97706RawTerms
def rightRaw : List Term := Proof.Events381.exact97683RawTerms
def group : MergeGroup := .operator 97706 97683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97706) (leftOrdinal := 1)
    (rightResult := 97683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34047⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97711

namespace LeftMerge97713
def owner : Owner := ⟨.program ⟨257⟩, ⟨34048⟩⟩
def mergeEvent : Nat := 97713
def frameStart : Nat := 97632
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33146⟩⟩] } }
def rhsRaw : List Term := Proof.Events381.exact97680RawTerms
def group : MergeGroup := .relation 97712
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97712) (rhsResult := 97680)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34047⟩⟩) ⟨33146⟩ 97680) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33146⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨33146⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97713

namespace LeftMerge97721
def owner : Owner := ⟨.program ⟨257⟩, ⟨32203⟩⟩
def mergeEvent : Nat := 97721
def frameStart : Nat := 97632
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97694RawTerms
def rightRaw : List Term := Proof.Events381.exact97717RawTerms
def group : MergeGroup := .operator 97694 97717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97694) (leftOrdinal := 0)
    (rightResult := 97717) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32201⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97721

namespace LeftMerge97738
def owner : Owner := ⟨.program ⟨257⟩, ⟨32799⟩⟩
def mergeEvent : Nat := 97738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }
def rhsRaw : List Term := Proof.Events381.exact97735RawTerms
def group : MergeGroup := .relation 97737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97737) (rhsResult := 97735)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97736 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32796⟩⟩]⟩) (none) 97735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97738

namespace LeftMerge97739
def owner : Owner := ⟨.program ⟨257⟩, ⟨32799⟩⟩
def mergeEvent : Nat := 97739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩] } }
def rhsRaw : List Term := Proof.Events381.exact97735RawTerms
def group : MergeGroup := .relation 97737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97737) (rhsResult := 97735)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97736 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32796⟩⟩]⟩) (none) 97735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34047⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97739

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
