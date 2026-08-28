import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge274306
def owner : Owner := ⟨.program ⟨257⟩, ⟨17269⟩⟩
def mergeEvent : Nat := 274306
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } }
def leftRaw : List Term := Proof.Events1071.exact274297RawTerms
def rightRaw : List Term := Proof.Events1071.exact274233RawTerms
def group : MergeGroup := .operator 274297 274233
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274297) (leftOrdinal := 0)
    (rightResult := 274233) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17268⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274306

namespace LeftMerge274320
def owner : Owner := ⟨.program ⟨257⟩, ⟨16209⟩⟩
def mergeEvent : Nat := 274320
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1071.exact274314RawTerms
def group : MergeGroup := .operator 266120 274314
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 274314) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16206⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274320

namespace LeftMerge274399
def owner : Owner := ⟨.program ⟨257⟩, ⟨15275⟩⟩
def mergeEvent : Nat := 274399
def frameStart : Nat := 274369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1071.exact274395RawTerms
def rightRaw : List Term := Proof.Events1071.exact274392RawTerms
def group : MergeGroup := .operator 274395 274392
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274395) (leftOrdinal := 0)
    (rightResult := 274392) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274399

namespace LeftMerge274429
def owner : Owner := ⟨.program ⟨257⟩, ⟨17096⟩⟩
def mergeEvent : Nat := 274429
def frameStart : Nat := 274369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1071.exact274425RawTerms
def rightRaw : List Term := Proof.Events1071.exact274423RawTerms
def group : MergeGroup := .operator 274425 274423
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274425) (leftOrdinal := 0)
    (rightResult := 274423) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274429

namespace LeftMerge274452
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 274452
def frameStart : Nat := 274369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events1072.exact274448RawTerms
def rightRaw : List Term := Proof.Events1072.exact274445RawTerms
def group : MergeGroup := .operator 274448 274445
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274448) (leftOrdinal := 0)
    (rightResult := 274445) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274452

namespace LeftMerge274461
def owner : Owner := ⟨.program ⟨257⟩, ⟨17271⟩⟩
def mergeEvent : Nat := 274461
def frameStart : Nat := 274369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } }
def leftRaw : List Term := Proof.Events1072.exact274457RawTerms
def rightRaw : List Term := Proof.Events1071.exact274414RawTerms
def group : MergeGroup := .operator 274457 274414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274457) (leftOrdinal := 0)
    (rightResult := 274414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17268⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274461

namespace LeftMerge274462
def owner : Owner := ⟨.program ⟨257⟩, ⟨17271⟩⟩
def mergeEvent : Nat := 274462
def frameStart : Nat := 274369
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } }
def leftRaw : List Term := Proof.Events1072.exact274457RawTerms
def rightRaw : List Term := Proof.Events1071.exact274414RawTerms
def group : MergeGroup := .operator 274457 274414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274457) (leftOrdinal := 1)
    (rightResult := 274414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17268⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274462

namespace LeftMerge274464
def owner : Owner := ⟨.program ⟨257⟩, ⟨17271⟩⟩
def mergeEvent : Nat := 274464
def frameStart : Nat := 274369
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16799⟩⟩] } }
def rhsRaw : List Term := Proof.Events1071.exact274411RawTerms
def group : MergeGroup := .relation 274463
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274463) (rhsResult := 274411)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17268⟩⟩) ⟨16799⟩ 274411) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16799⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], [⟨.program ⟨257⟩, ⟨16799⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274464

namespace LeftMerge274472
def owner : Owner := ⟨.program ⟨257⟩, ⟨15724⟩⟩
def mergeEvent : Nat := 274472
def frameStart : Nat := 274369
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1071.exact274425RawTerms
def rightRaw : List Term := Proof.Events1072.exact274468RawTerms
def group : MergeGroup := .operator 274425 274468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274425) (leftOrdinal := 0)
    (rightResult := 274468) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15722⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274472

namespace LeftMerge274489
def owner : Owner := ⟨.program ⟨257⟩, ⟨16209⟩⟩
def mergeEvent : Nat := 274489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events1072.exact274486RawTerms
def group : MergeGroup := .relation 274488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274488) (rhsResult := 274486)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 274487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩]⟩) (none) 274486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274489

namespace LeftMerge274490
def owner : Owner := ⟨.program ⟨257⟩, ⟨16209⟩⟩
def mergeEvent : Nat := 274490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } }
def rhsRaw : List Term := Proof.Events1072.exact274486RawTerms
def group : MergeGroup := .relation 274488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274488) (rhsResult := 274486)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 274487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩]⟩) (none) 274486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274490

namespace LeftMerge274491
def owner : Owner := ⟨.program ⟨257⟩, ⟨16209⟩⟩
def mergeEvent : Nat := 274491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16799⟩⟩] } }
def rhsRaw : List Term := Proof.Events1072.exact274486RawTerms
def group : MergeGroup := .relation 274488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274488) (rhsResult := 274486)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 274487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩]⟩) (none) 274486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16799⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], [⟨.program ⟨257⟩, ⟨16799⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274491

namespace LeftMerge274492
def owner : Owner := ⟨.program ⟨257⟩, ⟨16209⟩⟩
def mergeEvent : Nat := 274492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1072.exact274486RawTerms
def group : MergeGroup := .relation 274488
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274488) (rhsResult := 274486)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 274487 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16206⟩⟩]⟩) (none) 274486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15722⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨15722⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274492

namespace LeftMerge274497
def owner : Owner := ⟨.program ⟨257⟩, ⟨17270⟩⟩
def mergeEvent : Nat := 274497
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16799⟩⟩] } }
def leftRaw : List Term := Proof.Events1072.exact274493RawTerms
def rightRaw : List Term := Proof.Events1071.exact274307RawTerms
def group : MergeGroup := .operator 274493 274307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274493) (leftOrdinal := 2)
    (rightResult := 274307) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16799⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16799⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12256⟩⟩, ⟨.program ⟨257⟩, ⟨15274⟩⟩], [⟨.program ⟨257⟩, ⟨16799⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274497

namespace LeftMerge274498
def owner : Owner := ⟨.program ⟨257⟩, ⟨17270⟩⟩
def mergeEvent : Nat := 274498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } }
def leftRaw : List Term := Proof.Events1072.exact274493RawTerms
def rightRaw : List Term := Proof.Events1071.exact274307RawTerms
def group : MergeGroup := .operator 274493 274307
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274493) (leftOrdinal := 1)
    (rightResult := 274307) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17268⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274498

namespace LeftMerge274506
def owner : Owner := ⟨.program ⟨257⟩, ⟨17531⟩⟩
def mergeEvent : Nat := 274506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17529⟩⟩] } }
def leftRaw : List Term := Proof.Events1072.exact274500RawTerms
def rightRaw : List Term := Proof.Events1071.exact274223RawTerms
def group : MergeGroup := .operator 274500 274223
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274500) (leftOrdinal := 0)
    (rightResult := 274223) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17529⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274506

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
