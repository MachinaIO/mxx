import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge58622
def owner : Owner := ⟨.program ⟨214⟩, ⟨24996⟩⟩
def mergeEvent : Nat := 58622
def frameStart : Nat := 58529
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩] } }
def leftRaw : List Term := Proof.Events228.exact58617RawTerms
def rightRaw : List Term := Proof.Events228.exact58574RawTerms
def group : MergeGroup := .operator 58617 58574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58617) (leftOrdinal := 1)
    (rightResult := 58574) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24993⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58622

namespace LeftMerge58624
def owner : Owner := ⟨.program ⟨214⟩, ⟨24996⟩⟩
def mergeEvent : Nat := 58624
def frameStart : Nat := 58529
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22998⟩⟩] } }
def rhsRaw : List Term := Proof.Events228.exact58571RawTerms
def group : MergeGroup := .relation 58623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58623) (rhsResult := 58571)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24993⟩⟩) ⟨22998⟩ 58571) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22998⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], [⟨.program ⟨214⟩, ⟨22998⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58624

namespace LeftMerge58632
def owner : Owner := ⟨.program ⟨214⟩, ⟨14959⟩⟩
def mergeEvent : Nat := 58632
def frameStart : Nat := 58529
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events228.exact58585RawTerms
def rightRaw : List Term := Proof.Events229.exact58628RawTerms
def group : MergeGroup := .operator 58585 58628
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58585) (leftOrdinal := 0)
    (rightResult := 58628) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58632

namespace LeftMerge58649
def owner : Owner := ⟨.program ⟨214⟩, ⟨19103⟩⟩
def mergeEvent : Nat := 58649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }
def rhsRaw : List Term := Proof.Events229.exact58646RawTerms
def group : MergeGroup := .relation 58648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58648) (rhsResult := 58646)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19100⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58647 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19100⟩⟩]⟩) (none) 58646) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58649

namespace LeftMerge58650
def owner : Owner := ⟨.program ⟨214⟩, ⟨19103⟩⟩
def mergeEvent : Nat := 58650
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩] } }
def rhsRaw : List Term := Proof.Events229.exact58646RawTerms
def group : MergeGroup := .relation 58648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58648) (rhsResult := 58646)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19100⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58647 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19100⟩⟩]⟩) (none) 58646) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58650

namespace LeftMerge58651
def owner : Owner := ⟨.program ⟨214⟩, ⟨19103⟩⟩
def mergeEvent : Nat := 58651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22998⟩⟩] } }
def rhsRaw : List Term := Proof.Events229.exact58646RawTerms
def group : MergeGroup := .relation 58648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58648) (rhsResult := 58646)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19100⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58647 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19100⟩⟩]⟩) (none) 58646) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22998⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], [⟨.program ⟨214⟩, ⟨22998⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58651

namespace LeftMerge58652
def owner : Owner := ⟨.program ⟨214⟩, ⟨19103⟩⟩
def mergeEvent : Nat := 58652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events229.exact58646RawTerms
def group : MergeGroup := .relation 58648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58648) (rhsResult := 58646)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19100⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 58647 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19100⟩⟩]⟩) (none) 58646) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58652

namespace LeftMerge58657
def owner : Owner := ⟨.program ⟨214⟩, ⟨24995⟩⟩
def mergeEvent : Nat := 58657
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22998⟩⟩] } }
def leftRaw : List Term := Proof.Events229.exact58653RawTerms
def rightRaw : List Term := Proof.Events228.exact58467RawTerms
def group : MergeGroup := .operator 58653 58467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58653) (leftOrdinal := 2)
    (rightResult := 58467) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22998⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22998⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨9510⟩⟩, ⟨.program ⟨214⟩, ⟨10684⟩⟩], [⟨.program ⟨214⟩, ⟨22998⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58657

namespace LeftMerge58658
def owner : Owner := ⟨.program ⟨214⟩, ⟨24995⟩⟩
def mergeEvent : Nat := 58658
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩] } }
def leftRaw : List Term := Proof.Events229.exact58653RawTerms
def rightRaw : List Term := Proof.Events228.exact58467RawTerms
def group : MergeGroup := .operator 58653 58467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58653) (leftOrdinal := 1)
    (rightResult := 58467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨24993⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58658

namespace LeftMerge58666
def owner : Owner := ⟨.program ⟨214⟩, ⟨26579⟩⟩
def mergeEvent : Nat := 58666
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26577⟩⟩] } }
def leftRaw : List Term := Proof.Events229.exact58660RawTerms
def rightRaw : List Term := Proof.Events228.exact58383RawTerms
def group : MergeGroup := .operator 58660 58383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58660) (leftOrdinal := 0)
    (rightResult := 58383) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26577⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58666

namespace LeftMerge58667
def owner : Owner := ⟨.program ⟨214⟩, ⟨26579⟩⟩
def mergeEvent : Nat := 58667
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26577⟩⟩] } }
def leftRaw : List Term := Proof.Events229.exact58660RawTerms
def rightRaw : List Term := Proof.Events228.exact58383RawTerms
def group : MergeGroup := .operator 58660 58383
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58660) (leftOrdinal := 1)
    (rightResult := 58383) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26577⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26577⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58667

namespace LeftMerge58669
def owner : Owner := ⟨.program ⟨214⟩, ⟨26579⟩⟩
def mergeEvent : Nat := 58669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23787⟩⟩] } }
def rhsRaw : List Term := Proof.Events228.exact58380RawTerms
def group : MergeGroup := .relation 58668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 58668) (rhsResult := 58380)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26577⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26577⟩⟩) ⟨23787⟩ 58380) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23787⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨14957⟩⟩], [⟨.program ⟨214⟩, ⟨23787⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58669

namespace LeftMerge58683
def owner : Owner := ⟨.program ⟨214⟩, ⟨20543⟩⟩
def mergeEvent : Nat := 58683
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20540⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events229.exact58677RawTerms
def group : MergeGroup := .operator 50762 58677
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 58677) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20540⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20540⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58683

namespace LeftMerge58804
def owner : Owner := ⟨.program ⟨214⟩, ⟨14999⟩⟩
def mergeEvent : Nat := 58804
def frameStart : Nat := 58738
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events229.exact58800RawTerms
def rightRaw : List Term := Proof.Events229.exact58798RawTerms
def group : MergeGroup := .operator 58800 58798
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58800) (leftOrdinal := 0)
    (rightResult := 58798) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58804

namespace LeftMerge58816
def owner : Owner := ⟨.program ⟨214⟩, ⟨26578⟩⟩
def mergeEvent : Nat := 58816
def frameStart : Nat := 58738
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26577⟩⟩] } }
def leftRaw : List Term := Proof.Events229.exact58812RawTerms
def rightRaw : List Term := Proof.Events229.exact58789RawTerms
def group : MergeGroup := .operator 58812 58789
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58812) (leftOrdinal := 0)
    (rightResult := 58789) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge58816

namespace LeftMerge58817
def owner : Owner := ⟨.program ⟨214⟩, ⟨26578⟩⟩
def mergeEvent : Nat := 58817
def frameStart : Nat := 58738
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26577⟩⟩] } }
def leftRaw : List Term := Proof.Events229.exact58812RawTerms
def rightRaw : List Term := Proof.Events229.exact58789RawTerms
def group : MergeGroup := .operator 58812 58789
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 58812) (leftOrdinal := 1)
    (rightResult := 58789) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14957⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26577⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14957⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26577⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge58817

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
