import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge23598
def owner : Owner := ⟨.program ⟨214⟩, ⟨16563⟩⟩
def mergeEvent : Nat := 23598
def frameStart : Nat := 23495
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events091.exact23551RawTerms
def rightRaw : List Term := Proof.Events092.exact23594RawTerms
def group : MergeGroup := .operator 23551 23594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23551) (leftOrdinal := 0)
    (rightResult := 23594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16561⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23598

namespace LeftMerge23615
def owner : Owner := ⟨.program ⟨214⟩, ⟨19975⟩⟩
def mergeEvent : Nat := 23615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }
def rhsRaw : List Term := Proof.Events092.exact23612RawTerms
def group : MergeGroup := .relation 23614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23614) (rhsResult := 23612)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19972⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23613 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19972⟩⟩]⟩) (none) 23612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23615

namespace LeftMerge23616
def owner : Owner := ⟨.program ⟨214⟩, ⟨19975⟩⟩
def mergeEvent : Nat := 23616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩] } }
def rhsRaw : List Term := Proof.Events092.exact23612RawTerms
def group : MergeGroup := .relation 23614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23614) (rhsResult := 23612)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19972⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23613 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19972⟩⟩]⟩) (none) 23612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23616

namespace LeftMerge23617
def owner : Owner := ⟨.program ⟨214⟩, ⟨19975⟩⟩
def mergeEvent : Nat := 23617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩, ⟨.program ⟨214⟩, ⟨12590⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23254⟩⟩] } }
def rhsRaw : List Term := Proof.Events092.exact23612RawTerms
def group : MergeGroup := .relation 23614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23614) (rhsResult := 23612)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19972⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23613 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19972⟩⟩]⟩) (none) 23612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9940⟩⟩, ⟨.program ⟨214⟩, ⟨12590⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23254⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩, ⟨.program ⟨214⟩, ⟨12590⟩⟩], [⟨.program ⟨214⟩, ⟨23254⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23617

namespace LeftMerge23618
def owner : Owner := ⟨.program ⟨214⟩, ⟨19975⟩⟩
def mergeEvent : Nat := 23618
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events092.exact23612RawTerms
def group : MergeGroup := .relation 23614
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23614) (rhsResult := 23612)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19972⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 23613 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19972⟩⟩]⟩) (none) 23612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16561⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23618

namespace LeftMerge23623
def owner : Owner := ⟨.program ⟨214⟩, ⟨25467⟩⟩
def mergeEvent : Nat := 23623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩, ⟨.program ⟨214⟩, ⟨12590⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23254⟩⟩] } }
def leftRaw : List Term := Proof.Events092.exact23619RawTerms
def rightRaw : List Term := Proof.Events091.exact23433RawTerms
def group : MergeGroup := .operator 23619 23433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23619) (leftOrdinal := 2)
    (rightResult := 23433) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩, ⟨.program ⟨214⟩, ⟨12590⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23254⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩, ⟨.program ⟨214⟩, ⟨12590⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23254⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9940⟩⟩, ⟨.program ⟨214⟩, ⟨12590⟩⟩], [⟨.program ⟨214⟩, ⟨23254⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23623

namespace LeftMerge23624
def owner : Owner := ⟨.program ⟨214⟩, ⟨25467⟩⟩
def mergeEvent : Nat := 23624
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩] } }
def leftRaw : List Term := Proof.Events092.exact23619RawTerms
def rightRaw : List Term := Proof.Events091.exact23433RawTerms
def group : MergeGroup := .operator 23619 23433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23619) (leftOrdinal := 1)
    (rightResult := 23433) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25465⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23624

namespace LeftMerge23632
def owner : Owner := ⟨.program ⟨214⟩, ⟨29209⟩⟩
def mergeEvent : Nat := 23632
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩] } }
def leftRaw : List Term := Proof.Events092.exact23626RawTerms
def rightRaw : List Term := Proof.Events091.exact23349RawTerms
def group : MergeGroup := .operator 23626 23349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23626) (leftOrdinal := 0)
    (rightResult := 23349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29207⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23632

namespace LeftMerge23633
def owner : Owner := ⟨.program ⟨214⟩, ⟨29209⟩⟩
def mergeEvent : Nat := 23633
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩] } }
def leftRaw : List Term := Proof.Events092.exact23626RawTerms
def rightRaw : List Term := Proof.Events091.exact23349RawTerms
def group : MergeGroup := .operator 23626 23349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23626) (leftOrdinal := 1)
    (rightResult := 23349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29207⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16561⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23633

namespace LeftMerge23635
def owner : Owner := ⟨.program ⟨214⟩, ⟨29209⟩⟩
def mergeEvent : Nat := 23635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24549⟩⟩] } }
def rhsRaw : List Term := Proof.Events091.exact23346RawTerms
def group : MergeGroup := .relation 23634
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23634) (rhsResult := 23346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16561⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29207⟩⟩) ⟨24549⟩ 23346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24549⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16561⟩⟩], [⟨.program ⟨214⟩, ⟨24549⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23635

namespace LeftMerge23649
def owner : Owner := ⟨.program ⟨214⟩, ⟨22279⟩⟩
def mergeEvent : Nat := 23649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22276⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events092.exact23643RawTerms
def group : MergeGroup := .operator 21512 23643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 23643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22276⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23649

namespace LeftMerge23770
def owner : Owner := ⟨.program ⟨214⟩, ⟨16603⟩⟩
def mergeEvent : Nat := 23770
def frameStart : Nat := 23704
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events092.exact23766RawTerms
def rightRaw : List Term := Proof.Events092.exact23764RawTerms
def group : MergeGroup := .operator 23766 23764
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23766) (leftOrdinal := 0)
    (rightResult := 23764) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16561⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23770

namespace LeftMerge23782
def owner : Owner := ⟨.program ⟨214⟩, ⟨29208⟩⟩
def mergeEvent : Nat := 23782
def frameStart : Nat := 23704
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩] } }
def leftRaw : List Term := Proof.Events092.exact23778RawTerms
def rightRaw : List Term := Proof.Events092.exact23755RawTerms
def group : MergeGroup := .operator 23778 23755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23778) (leftOrdinal := 0)
    (rightResult := 23755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6703⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29207⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6703⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23782

namespace LeftMerge23783
def owner : Owner := ⟨.program ⟨214⟩, ⟨29208⟩⟩
def mergeEvent : Nat := 23783
def frameStart : Nat := 23704
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩] } }
def leftRaw : List Term := Proof.Events092.exact23778RawTerms
def rightRaw : List Term := Proof.Events092.exact23755RawTerms
def group : MergeGroup := .operator 23778 23755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23778) (leftOrdinal := 1)
    (rightResult := 23755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29207⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16561⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23783

namespace LeftMerge23785
def owner : Owner := ⟨.program ⟨214⟩, ⟨29208⟩⟩
def mergeEvent : Nat := 23785
def frameStart : Nat := 23704
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24549⟩⟩] } }
def rhsRaw : List Term := Proof.Events092.exact23752RawTerms
def group : MergeGroup := .relation 23784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 23784) (rhsResult := 23752)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16561⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29207⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29207⟩⟩) ⟨24549⟩ 23752) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24549⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16561⟩⟩], [⟨.program ⟨214⟩, ⟨24549⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge23785

namespace LeftMerge23793
def owner : Owner := ⟨.program ⟨214⟩, ⟨18215⟩⟩
def mergeEvent : Nat := 23793
def frameStart : Nat := 23704
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18214⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events092.exact23766RawTerms
def rightRaw : List Term := Proof.Events092.exact23789RawTerms
def group : MergeGroup := .operator 23766 23789
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 23766) (leftOrdinal := 0)
    (rightResult := 23789) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18214⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18214⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge23793

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
