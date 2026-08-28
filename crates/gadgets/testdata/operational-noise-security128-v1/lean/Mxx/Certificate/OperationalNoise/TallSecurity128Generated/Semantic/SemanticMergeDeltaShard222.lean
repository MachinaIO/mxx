import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge40306
def owner : Owner := ⟨.program ⟨257⟩, ⟨17459⟩⟩
def mergeEvent : Nat := 40306
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40297RawTerms
def rightRaw : List Term := Proof.Events157.exact40233RawTerms
def group : MergeGroup := .operator 40297 40233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40297) (leftOrdinal := 0)
    (rightResult := 40233) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17458⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40306

namespace LeftMerge40320
def owner : Owner := ⟨.program ⟨257⟩, ⟨16382⟩⟩
def mergeEvent : Nat := 40320
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events157.exact40314RawTerms
def group : MergeGroup := .operator 32120 40314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 40314) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16379⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40320

namespace LeftMerge40399
def owner : Owner := ⟨.program ⟨257⟩, ⟨15691⟩⟩
def mergeEvent : Nat := 40399
def frameStart : Nat := 40369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events157.exact40395RawTerms
def rightRaw : List Term := Proof.Events157.exact40392RawTerms
def group : MergeGroup := .operator 40395 40392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40395) (leftOrdinal := 0)
    (rightResult := 40392) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12516⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40399

namespace LeftMerge40429
def owner : Owner := ⟨.program ⟨257⟩, ⟨17164⟩⟩
def mergeEvent : Nat := 40429
def frameStart : Nat := 40369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40425RawTerms
def rightRaw : List Term := Proof.Events157.exact40423RawTerms
def group : MergeGroup := .operator 40425 40423
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40425) (leftOrdinal := 0)
    (rightResult := 40423) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40429

namespace LeftMerge40452
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 40452
def frameStart : Nat := 40369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events158.exact40448RawTerms
def rightRaw : List Term := Proof.Events157.exact40445RawTerms
def group : MergeGroup := .operator 40448 40445
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40448) (leftOrdinal := 0)
    (rightResult := 40445) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40452

namespace LeftMerge40461
def owner : Owner := ⟨.program ⟨257⟩, ⟨17461⟩⟩
def mergeEvent : Nat := 40461
def frameStart : Nat := 40369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩] } }
def leftRaw : List Term := Proof.Events158.exact40457RawTerms
def rightRaw : List Term := Proof.Events157.exact40414RawTerms
def group : MergeGroup := .operator 40457 40414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40457) (leftOrdinal := 0)
    (rightResult := 40414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17458⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40461

namespace LeftMerge40462
def owner : Owner := ⟨.program ⟨257⟩, ⟨17461⟩⟩
def mergeEvent : Nat := 40462
def frameStart : Nat := 40369
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩] } }
def leftRaw : List Term := Proof.Events158.exact40457RawTerms
def rightRaw : List Term := Proof.Events157.exact40414RawTerms
def group : MergeGroup := .operator 40457 40414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40457) (leftOrdinal := 1)
    (rightResult := 40414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17458⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40462

namespace LeftMerge40464
def owner : Owner := ⟨.program ⟨257⟩, ⟨17461⟩⟩
def mergeEvent : Nat := 40464
def frameStart : Nat := 40369
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16903⟩⟩] } }
def rhsRaw : List Term := Proof.Events157.exact40411RawTerms
def group : MergeGroup := .relation 40463
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40463) (rhsResult := 40411)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17458⟩⟩) ⟨16903⟩ 40411) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16903⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], [⟨.program ⟨257⟩, ⟨16903⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40464

namespace LeftMerge40472
def owner : Owner := ⟨.program ⟨257⟩, ⟨15862⟩⟩
def mergeEvent : Nat := 40472
def frameStart : Nat := 40369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40425RawTerms
def rightRaw : List Term := Proof.Events158.exact40468RawTerms
def group : MergeGroup := .operator 40425 40468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40425) (leftOrdinal := 0)
    (rightResult := 40468) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40472

namespace LeftMerge40489
def owner : Owner := ⟨.program ⟨257⟩, ⟨16382⟩⟩
def mergeEvent : Nat := 40489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events158.exact40486RawTerms
def group : MergeGroup := .relation 40488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40488) (rhsResult := 40486)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩]⟩) (none) 40486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40489

namespace LeftMerge40490
def owner : Owner := ⟨.program ⟨257⟩, ⟨16382⟩⟩
def mergeEvent : Nat := 40490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩] } }
def rhsRaw : List Term := Proof.Events158.exact40486RawTerms
def group : MergeGroup := .relation 40488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40488) (rhsResult := 40486)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩]⟩) (none) 40486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40490

namespace LeftMerge40491
def owner : Owner := ⟨.program ⟨257⟩, ⟨16382⟩⟩
def mergeEvent : Nat := 40491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16903⟩⟩] } }
def rhsRaw : List Term := Proof.Events158.exact40486RawTerms
def group : MergeGroup := .relation 40488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40488) (rhsResult := 40486)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩]⟩) (none) 40486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16903⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], [⟨.program ⟨257⟩, ⟨16903⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40491

namespace LeftMerge40492
def owner : Owner := ⟨.program ⟨257⟩, ⟨16382⟩⟩
def mergeEvent : Nat := 40492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events158.exact40486RawTerms
def group : MergeGroup := .relation 40488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40488) (rhsResult := 40486)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16379⟩⟩]⟩) (none) 40486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨15860⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40492

namespace LeftMerge40497
def owner : Owner := ⟨.program ⟨257⟩, ⟨17460⟩⟩
def mergeEvent : Nat := 40497
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16903⟩⟩] } }
def leftRaw : List Term := Proof.Events158.exact40493RawTerms
def rightRaw : List Term := Proof.Events157.exact40307RawTerms
def group : MergeGroup := .operator 40493 40307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40493) (leftOrdinal := 2)
    (rightResult := 40307) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16903⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16903⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨12516⟩⟩, ⟨.program ⟨257⟩, ⟨15690⟩⟩], [⟨.program ⟨257⟩, ⟨16903⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40497

namespace LeftMerge40498
def owner : Owner := ⟨.program ⟨257⟩, ⟨17460⟩⟩
def mergeEvent : Nat := 40498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩] } }
def leftRaw : List Term := Proof.Events158.exact40493RawTerms
def rightRaw : List Term := Proof.Events157.exact40307RawTerms
def group : MergeGroup := .operator 40493 40307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40493) (leftOrdinal := 1)
    (rightResult := 40307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17458⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40498

namespace LeftMerge40506
def owner : Owner := ⟨.program ⟨257⟩, ⟨18015⟩⟩
def mergeEvent : Nat := 40506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨18013⟩⟩] } }
def leftRaw : List Term := Proof.Events158.exact40500RawTerms
def rightRaw : List Term := Proof.Events157.exact40223RawTerms
def group : MergeGroup := .operator 40500 40223
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40500) (leftOrdinal := 0)
    (rightResult := 40223) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨18013⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨18013⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40506

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
