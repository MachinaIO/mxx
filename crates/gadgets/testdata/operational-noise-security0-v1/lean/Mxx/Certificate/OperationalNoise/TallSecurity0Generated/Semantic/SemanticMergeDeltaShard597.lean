import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge96756
def owner : Owner := ⟨.program ⟨214⟩, ⟨25363⟩⟩
def mergeEvent : Nat := 96756
def frameStart : Nat := 96675
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96751RawTerms
def rightRaw : List Term := Proof.Events377.exact96708RawTerms
def group : MergeGroup := .operator 96751 96708
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96751) (leftOrdinal := 1)
    (rightResult := 96708) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25360⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96756

namespace LeftMerge96758
def owner : Owner := ⟨.program ⟨214⟩, ⟨25363⟩⟩
def mergeEvent : Nat := 96758
def frameStart : Nat := 96675
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23200⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96705RawTerms
def group : MergeGroup := .relation 96757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96757) (rhsResult := 96705)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25360⟩⟩) ⟨23200⟩ 96705) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23200⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨23200⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96758

namespace LeftMerge96766
def owner : Owner := ⟨.program ⟨214⟩, ⟨16457⟩⟩
def mergeEvent : Nat := 96766
def frameStart : Nat := 96675
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events377.exact96719RawTerms
def rightRaw : List Term := Proof.Events377.exact96762RawTerms
def group : MergeGroup := .operator 96719 96762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96719) (leftOrdinal := 0)
    (rightResult := 96762) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96766

namespace LeftMerge96783
def owner : Owner := ⟨.program ⟨214⟩, ⟨19880⟩⟩
def mergeEvent : Nat := 96783
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }
def rhsRaw : List Term := Proof.Events378.exact96780RawTerms
def group : MergeGroup := .relation 96782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96782) (rhsResult := 96780)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96781 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩) (none) 96780) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96783

namespace LeftMerge96784
def owner : Owner := ⟨.program ⟨214⟩, ⟨19880⟩⟩
def mergeEvent : Nat := 96784
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩] } }
def rhsRaw : List Term := Proof.Events378.exact96780RawTerms
def group : MergeGroup := .relation 96782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96782) (rhsResult := 96780)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96781 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩) (none) 96780) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96784

namespace LeftMerge96785
def owner : Owner := ⟨.program ⟨214⟩, ⟨19880⟩⟩
def mergeEvent : Nat := 96785
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23200⟩⟩] } }
def rhsRaw : List Term := Proof.Events378.exact96780RawTerms
def group : MergeGroup := .relation 96782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96782) (rhsResult := 96780)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96781 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩) (none) 96780) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23200⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨23200⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96785

namespace LeftMerge96786
def owner : Owner := ⟨.program ⟨214⟩, ⟨19880⟩⟩
def mergeEvent : Nat := 96786
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events378.exact96780RawTerms
def group : MergeGroup := .relation 96782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96782) (rhsResult := 96780)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 96781 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19877⟩⟩]⟩) (none) 96780) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96786

namespace LeftMerge96791
def owner : Owner := ⟨.program ⟨214⟩, ⟨25362⟩⟩
def mergeEvent : Nat := 96791
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23200⟩⟩] } }
def leftRaw : List Term := Proof.Events378.exact96787RawTerms
def rightRaw : List Term := Proof.Events377.exact96625RawTerms
def group : MergeGroup := .operator 96787 96625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96787) (leftOrdinal := 2)
    (rightResult := 96625) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23200⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23200⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨9805⟩⟩, ⟨.program ⟨214⟩, ⟨12346⟩⟩], [⟨.program ⟨214⟩, ⟨23200⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96791

namespace LeftMerge96792
def owner : Owner := ⟨.program ⟨214⟩, ⟨25362⟩⟩
def mergeEvent : Nat := 96792
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩] } }
def leftRaw : List Term := Proof.Events378.exact96787RawTerms
def rightRaw : List Term := Proof.Events377.exact96625RawTerms
def group : MergeGroup := .operator 96787 96625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96787) (leftOrdinal := 1)
    (rightResult := 96625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25360⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96792

namespace LeftMerge96800
def owner : Owner := ⟨.program ⟨214⟩, ⟨28918⟩⟩
def mergeEvent : Nat := 96800
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩] } }
def leftRaw : List Term := Proof.Events378.exact96794RawTerms
def rightRaw : List Term := Proof.Events377.exact96541RawTerms
def group : MergeGroup := .operator 96794 96541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96794) (leftOrdinal := 0)
    (rightResult := 96541) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28916⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96800

namespace LeftMerge96801
def owner : Owner := ⟨.program ⟨214⟩, ⟨28918⟩⟩
def mergeEvent : Nat := 96801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩] } }
def leftRaw : List Term := Proof.Events378.exact96794RawTerms
def rightRaw : List Term := Proof.Events377.exact96541RawTerms
def group : MergeGroup := .operator 96794 96541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96794) (leftOrdinal := 1)
    (rightResult := 96541) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28916⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96801

namespace LeftMerge96803
def owner : Owner := ⟨.program ⟨214⟩, ⟨28918⟩⟩
def mergeEvent : Nat := 96803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24468⟩⟩] } }
def rhsRaw : List Term := Proof.Events377.exact96538RawTerms
def group : MergeGroup := .relation 96802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 96802) (rhsResult := 96538)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28916⟩⟩) ⟨24468⟩ 96538) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24468⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨24468⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96803

namespace LeftMerge96817
def owner : Owner := ⟨.program ⟨214⟩, ⟨22112⟩⟩
def mergeEvent : Nat := 96817
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22109⟩⟩] } }
def leftRaw : List Term := Proof.Events368.exact94462RawTerms
def rightRaw : List Term := Proof.Events378.exact96811RawTerms
def group : MergeGroup := .operator 94462 96811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94462) (leftOrdinal := 0)
    (rightResult := 96811) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22109⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22109⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96817

namespace LeftMerge96914
def owner : Owner := ⟨.program ⟨214⟩, ⟨16499⟩⟩
def mergeEvent : Nat := 96914
def frameStart : Nat := 96860
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events378.exact96910RawTerms
def rightRaw : List Term := Proof.Events378.exact96908RawTerms
def group : MergeGroup := .operator 96910 96908
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96910) (leftOrdinal := 0)
    (rightResult := 96908) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96914

namespace LeftMerge96926
def owner : Owner := ⟨.program ⟨214⟩, ⟨28917⟩⟩
def mergeEvent : Nat := 96926
def frameStart : Nat := 96860
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩] } }
def leftRaw : List Term := Proof.Events378.exact96922RawTerms
def rightRaw : List Term := Proof.Events378.exact96899RawTerms
def group : MergeGroup := .operator 96922 96899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96922) (leftOrdinal := 0)
    (rightResult := 96899) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28916⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge96926

namespace LeftMerge96927
def owner : Owner := ⟨.program ⟨214⟩, ⟨28917⟩⟩
def mergeEvent : Nat := 96927
def frameStart : Nat := 96860
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩] } }
def leftRaw : List Term := Proof.Events378.exact96922RawTerms
def rightRaw : List Term := Proof.Events378.exact96899RawTerms
def group : MergeGroup := .operator 96922 96899
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96922) (leftOrdinal := 1)
    (rightResult := 96899) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16455⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28916⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16455⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28916⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge96927

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
