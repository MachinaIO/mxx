import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge44506
def owner : Owner := ⟨.program ⟨214⟩, ⟨19035⟩⟩
def mergeEvent : Nat := 44506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }
def rhsRaw : List Term := Proof.Events173.exact44503RawTerms
def group : MergeGroup := .relation 44505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44505) (rhsResult := 44503)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 44504 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩) (none) 44503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44506

namespace LeftMerge44507
def owner : Owner := ⟨.program ⟨214⟩, ⟨19035⟩⟩
def mergeEvent : Nat := 44507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩] } }
def rhsRaw : List Term := Proof.Events173.exact44503RawTerms
def group : MergeGroup := .relation 44505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44505) (rhsResult := 44503)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 44504 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩) (none) 44503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44507

namespace LeftMerge44508
def owner : Owner := ⟨.program ⟨214⟩, ⟨19035⟩⟩
def mergeEvent : Nat := 44508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22958⟩⟩] } }
def rhsRaw : List Term := Proof.Events173.exact44503RawTerms
def group : MergeGroup := .relation 44505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44505) (rhsResult := 44503)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 44504 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩) (none) 44503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22958⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨22958⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44508

namespace LeftMerge44509
def owner : Owner := ⟨.program ⟨214⟩, ⟨19035⟩⟩
def mergeEvent : Nat := 44509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events173.exact44503RawTerms
def group : MergeGroup := .relation 44505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44505) (rhsResult := 44503)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 44504 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩) (none) 44503) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44509

namespace LeftMerge44514
def owner : Owner := ⟨.program ⟨214⟩, ⟨24923⟩⟩
def mergeEvent : Nat := 44514
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22958⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44510RawTerms
def rightRaw : List Term := Proof.Events173.exact44324RawTerms
def group : MergeGroup := .operator 44510 44324
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44510) (leftOrdinal := 2)
    (rightResult := 44324) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22958⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22958⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9410⟩⟩, ⟨.program ⟨214⟩, ⟨10496⟩⟩], [⟨.program ⟨214⟩, ⟨22958⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44514

namespace LeftMerge44515
def owner : Owner := ⟨.program ⟨214⟩, ⟨24923⟩⟩
def mergeEvent : Nat := 44515
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44510RawTerms
def rightRaw : List Term := Proof.Events173.exact44324RawTerms
def group : MergeGroup := .operator 44510 44324
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44510) (leftOrdinal := 1)
    (rightResult := 44324) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44515

namespace LeftMerge44523
def owner : Owner := ⟨.program ⟨214⟩, ⟨26384⟩⟩
def mergeEvent : Nat := 44523
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44517RawTerms
def rightRaw : List Term := Proof.Events172.exact44240RawTerms
def group : MergeGroup := .operator 44517 44240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44517) (leftOrdinal := 0)
    (rightResult := 44240) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26382⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44523

namespace LeftMerge44524
def owner : Owner := ⟨.program ⟨214⟩, ⟨26384⟩⟩
def mergeEvent : Nat := 44524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44517RawTerms
def rightRaw : List Term := Proof.Events172.exact44240RawTerms
def group : MergeGroup := .operator 44517 44240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44517) (leftOrdinal := 1)
    (rightResult := 44240) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26382⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44524

namespace LeftMerge44526
def owner : Owner := ⟨.program ⟨214⟩, ⟨26384⟩⟩
def mergeEvent : Nat := 44526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23727⟩⟩] } }
def rhsRaw : List Term := Proof.Events172.exact44237RawTerms
def group : MergeGroup := .relation 44525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44525) (rhsResult := 44237)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26382⟩⟩) ⟨23727⟩ 44237) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23727⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44526

namespace LeftMerge44540
def owner : Owner := ⟨.program ⟨214⟩, ⟨20403⟩⟩
def mergeEvent : Nat := 44540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events173.exact44534RawTerms
def group : MergeGroup := .operator 36137 44534
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 44534) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20400⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44540

namespace LeftMerge44661
def owner : Owner := ⟨.program ⟨214⟩, ⟨14842⟩⟩
def mergeEvent : Nat := 44661
def frameStart : Nat := 44595
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events174.exact44657RawTerms
def rightRaw : List Term := Proof.Events174.exact44655RawTerms
def group : MergeGroup := .operator 44657 44655
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44657) (leftOrdinal := 0)
    (rightResult := 44655) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44661

namespace LeftMerge44673
def owner : Owner := ⟨.program ⟨214⟩, ⟨26383⟩⟩
def mergeEvent : Nat := 44673
def frameStart : Nat := 44595
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩] } }
def leftRaw : List Term := Proof.Events174.exact44669RawTerms
def rightRaw : List Term := Proof.Events174.exact44646RawTerms
def group : MergeGroup := .operator 44669 44646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44669) (leftOrdinal := 0)
    (rightResult := 44646) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26382⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44673

namespace LeftMerge44674
def owner : Owner := ⟨.program ⟨214⟩, ⟨26383⟩⟩
def mergeEvent : Nat := 44674
def frameStart : Nat := 44595
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩] } }
def leftRaw : List Term := Proof.Events174.exact44669RawTerms
def rightRaw : List Term := Proof.Events174.exact44646RawTerms
def group : MergeGroup := .operator 44669 44646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44669) (leftOrdinal := 1)
    (rightResult := 44646) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26382⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44674

namespace LeftMerge44676
def owner : Owner := ⟨.program ⟨214⟩, ⟨26383⟩⟩
def mergeEvent : Nat := 44676
def frameStart : Nat := 44595
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23727⟩⟩] } }
def rhsRaw : List Term := Proof.Events174.exact44643RawTerms
def group : MergeGroup := .relation 44675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44675) (rhsResult := 44643)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26382⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26382⟩⟩) ⟨23727⟩ 44643) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23727⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23727⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge44676

namespace LeftMerge44684
def owner : Owner := ⟨.program ⟨214⟩, ⟨15272⟩⟩
def mergeEvent : Nat := 44684
def frameStart : Nat := 44595
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events174.exact44657RawTerms
def rightRaw : List Term := Proof.Events174.exact44680RawTerms
def group : MergeGroup := .operator 44657 44680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44657) (leftOrdinal := 0)
    (rightResult := 44680) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15271⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15271⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44684

namespace LeftMerge44701
def owner : Owner := ⟨.program ⟨214⟩, ⟨20403⟩⟩
def mergeEvent : Nat := 44701
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }
def rhsRaw : List Term := Proof.Events174.exact44698RawTerms
def group : MergeGroup := .relation 44700
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 44700) (rhsResult := 44698)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 44699 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20400⟩⟩]⟩) (none) 44698) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge44701

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
