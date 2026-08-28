import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge83518
def owner : Owner := ⟨.program ⟨214⟩, ⟨25145⟩⟩
def mergeEvent : Nat := 83518
def frameStart : Nat := 83427
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83513RawTerms
def rightRaw : List Term := Proof.Events326.exact83472RawTerms
def group : MergeGroup := .operator 83513 83472
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83513) (leftOrdinal := 1)
    (rightResult := 83472) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25142⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83518

namespace LeftMerge83520
def owner : Owner := ⟨.program ⟨214⟩, ⟨25145⟩⟩
def mergeEvent : Nat := 83520
def frameStart : Nat := 83427
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23080⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83469RawTerms
def group : MergeGroup := .relation 83519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83519) (rhsResult := 83469)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25142⟩⟩) ⟨23080⟩ 83469) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23080⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83520

namespace LeftMerge83528
def owner : Owner := ⟨.program ⟨214⟩, ⟨16264⟩⟩
def mergeEvent : Nat := 83528
def frameStart : Nat := 83427
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83483RawTerms
def rightRaw : List Term := Proof.Events326.exact83524RawTerms
def group : MergeGroup := .operator 83483 83524
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83483) (leftOrdinal := 0)
    (rightResult := 83524) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83528

namespace LeftMerge83545
def owner : Owner := ⟨.program ⟨214⟩, ⟨19747⟩⟩
def mergeEvent : Nat := 83545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83542RawTerms
def group : MergeGroup := .relation 83544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83544) (rhsResult := 83542)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83543 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩) (none) 83542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83545

namespace LeftMerge83546
def owner : Owner := ⟨.program ⟨214⟩, ⟨19747⟩⟩
def mergeEvent : Nat := 83546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83542RawTerms
def group : MergeGroup := .relation 83544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83544) (rhsResult := 83542)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83543 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩) (none) 83542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83546

namespace LeftMerge83547
def owner : Owner := ⟨.program ⟨214⟩, ⟨19747⟩⟩
def mergeEvent : Nat := 83547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23080⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83542RawTerms
def group : MergeGroup := .relation 83544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83544) (rhsResult := 83542)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83543 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩) (none) 83542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23080⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83547

namespace LeftMerge83548
def owner : Owner := ⟨.program ⟨214⟩, ⟨19747⟩⟩
def mergeEvent : Nat := 83548
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83542RawTerms
def group : MergeGroup := .relation 83544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83544) (rhsResult := 83542)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83543 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19744⟩⟩]⟩) (none) 83542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83548

namespace LeftMerge83553
def owner : Owner := ⟨.program ⟨214⟩, ⟨25144⟩⟩
def mergeEvent : Nat := 83553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23080⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83549RawTerms
def rightRaw : List Term := Proof.Events325.exact83365RawTerms
def group : MergeGroup := .operator 83549 83365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83549) (leftOrdinal := 2)
    (rightResult := 83365) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23080⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23080⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨9610⟩⟩, ⟨.program ⟨214⟩, ⟨11761⟩⟩], [⟨.program ⟨214⟩, ⟨23080⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83553

namespace LeftMerge83554
def owner : Owner := ⟨.program ⟨214⟩, ⟨25144⟩⟩
def mergeEvent : Nat := 83554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83549RawTerms
def rightRaw : List Term := Proof.Events325.exact83365RawTerms
def group : MergeGroup := .operator 83549 83365
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83549) (leftOrdinal := 1)
    (rightResult := 83365) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩, ⟨.program ⟨214⟩, ⟨7861⟩⟩, ⟨.program ⟨214⟩, ⟨25142⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83554

namespace LeftMerge83562
def owner : Owner := ⟨.program ⟨214⟩, ⟨28519⟩⟩
def mergeEvent : Nat := 83562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83556RawTerms
def rightRaw : List Term := Proof.Events325.exact83281RawTerms
def group : MergeGroup := .operator 83556 83281
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83556) (leftOrdinal := 0)
    (rightResult := 83281) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28517⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83562

namespace LeftMerge83563
def owner : Owner := ⟨.program ⟨214⟩, ⟨28519⟩⟩
def mergeEvent : Nat := 83563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83556RawTerms
def rightRaw : List Term := Proof.Events325.exact83281RawTerms
def group : MergeGroup := .operator 83556 83281
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83556) (leftOrdinal := 1)
    (rightResult := 83281) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28517⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83563

namespace LeftMerge83565
def owner : Owner := ⟨.program ⟨214⟩, ⟨28519⟩⟩
def mergeEvent : Nat := 83565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24351⟩⟩] } }
def rhsRaw : List Term := Proof.Events325.exact83278RawTerms
def group : MergeGroup := .relation 83564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83564) (rhsResult := 83278)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28517⟩⟩) ⟨24351⟩ 83278) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24351⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨24351⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83565

namespace LeftMerge83579
def owner : Owner := ⟨.program ⟨214⟩, ⟨21835⟩⟩
def mergeEvent : Nat := 83579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21832⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events326.exact83573RawTerms
def group : MergeGroup := .operator 80012 83573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 83573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21832⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21832⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83579

namespace LeftMerge83700
def owner : Owner := ⟨.program ⟨214⟩, ⟨16339⟩⟩
def mergeEvent : Nat := 83700
def frameStart : Nat := 83634
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83696RawTerms
def rightRaw : List Term := Proof.Events326.exact83694RawTerms
def group : MergeGroup := .operator 83696 83694
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83696) (leftOrdinal := 0)
    (rightResult := 83694) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83700

namespace LeftMerge83712
def owner : Owner := ⟨.program ⟨214⟩, ⟨28518⟩⟩
def mergeEvent : Nat := 83712
def frameStart : Nat := 83634
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83708RawTerms
def rightRaw : List Term := Proof.Events326.exact83685RawTerms
def group : MergeGroup := .operator 83708 83685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83708) (leftOrdinal := 0)
    (rightResult := 83685) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28517⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83712

namespace LeftMerge83713
def owner : Owner := ⟨.program ⟨214⟩, ⟨28518⟩⟩
def mergeEvent : Nat := 83713
def frameStart : Nat := 83634
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83708RawTerms
def rightRaw : List Term := Proof.Events326.exact83685RawTerms
def group : MergeGroup := .operator 83708 83685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83708) (leftOrdinal := 1)
    (rightResult := 83685) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28517⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28517⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83713

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
