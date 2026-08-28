import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge66498
def owner : Owner := ⟨.program ⟨214⟩, ⟨25602⟩⟩
def mergeEvent : Nat := 66498
def frameStart : Nat := 66406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66494RawTerms
def rightRaw : List Term := Proof.Events259.exact66451RawTerms
def group : MergeGroup := .operator 66494 66451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66494) (leftOrdinal := 0)
    (rightResult := 66451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25599⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66498

namespace LeftMerge66499
def owner : Owner := ⟨.program ⟨214⟩, ⟨25602⟩⟩
def mergeEvent : Nat := 66499
def frameStart : Nat := 66406
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66494RawTerms
def rightRaw : List Term := Proof.Events259.exact66451RawTerms
def group : MergeGroup := .operator 66494 66451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66494) (leftOrdinal := 1)
    (rightResult := 66451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25599⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66499

namespace LeftMerge66501
def owner : Owner := ⟨.program ⟨214⟩, ⟨25602⟩⟩
def mergeEvent : Nat := 66501
def frameStart : Nat := 66406
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23330⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66448RawTerms
def group : MergeGroup := .relation 66500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66500) (rhsResult := 66448)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25599⟩⟩) ⟨23330⟩ 66448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23330⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], [⟨.program ⟨214⟩, ⟨23330⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66501

namespace LeftMerge66509
def owner : Owner := ⟨.program ⟨214⟩, ⟨16750⟩⟩
def mergeEvent : Nat := 66509
def frameStart : Nat := 66406
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66462RawTerms
def rightRaw : List Term := Proof.Events259.exact66505RawTerms
def group : MergeGroup := .operator 66462 66505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66462) (leftOrdinal := 0)
    (rightResult := 66505) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66509

namespace LeftMerge66526
def owner : Owner := ⟨.program ⟨214⟩, ⟨20103⟩⟩
def mergeEvent : Nat := 66526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66523RawTerms
def group : MergeGroup := .relation 66525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66525) (rhsResult := 66523)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20100⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66524 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20100⟩⟩]⟩) (none) 66523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66526

namespace LeftMerge66527
def owner : Owner := ⟨.program ⟨214⟩, ⟨20103⟩⟩
def mergeEvent : Nat := 66527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66523RawTerms
def group : MergeGroup := .relation 66525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66525) (rhsResult := 66523)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20100⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66524 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20100⟩⟩]⟩) (none) 66523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66527

namespace LeftMerge66528
def owner : Owner := ⟨.program ⟨214⟩, ⟨20103⟩⟩
def mergeEvent : Nat := 66528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23330⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66523RawTerms
def group : MergeGroup := .relation 66525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66525) (rhsResult := 66523)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20100⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66524 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20100⟩⟩]⟩) (none) 66523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23330⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], [⟨.program ⟨214⟩, ⟨23330⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66528

namespace LeftMerge66529
def owner : Owner := ⟨.program ⟨214⟩, ⟨20103⟩⟩
def mergeEvent : Nat := 66529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66523RawTerms
def group : MergeGroup := .relation 66525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66525) (rhsResult := 66523)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20100⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66524 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20100⟩⟩]⟩) (none) 66523) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66529

namespace LeftMerge66534
def owner : Owner := ⟨.program ⟨214⟩, ⟨25601⟩⟩
def mergeEvent : Nat := 66534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23330⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66530RawTerms
def rightRaw : List Term := Proof.Events259.exact66344RawTerms
def group : MergeGroup := .operator 66530 66344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66530) (leftOrdinal := 2)
    (rightResult := 66344) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23330⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23330⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10130⟩⟩, ⟨.program ⟨214⟩, ⟨12950⟩⟩], [⟨.program ⟨214⟩, ⟨23330⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66534

namespace LeftMerge66535
def owner : Owner := ⟨.program ⟨214⟩, ⟨25601⟩⟩
def mergeEvent : Nat := 66535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66530RawTerms
def rightRaw : List Term := Proof.Events259.exact66344RawTerms
def group : MergeGroup := .operator 66530 66344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66530) (leftOrdinal := 1)
    (rightResult := 66344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25599⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66535

namespace LeftMerge66543
def owner : Owner := ⟨.program ⟨214⟩, ⟨29591⟩⟩
def mergeEvent : Nat := 66543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66537RawTerms
def rightRaw : List Term := Proof.Events258.exact66260RawTerms
def group : MergeGroup := .operator 66537 66260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66537) (leftOrdinal := 0)
    (rightResult := 66260) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29589⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66543

namespace LeftMerge66544
def owner : Owner := ⟨.program ⟨214⟩, ⟨29591⟩⟩
def mergeEvent : Nat := 66544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66537RawTerms
def rightRaw : List Term := Proof.Events258.exact66260RawTerms
def group : MergeGroup := .operator 66537 66260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66537) (leftOrdinal := 1)
    (rightResult := 66260) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29589⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66544

namespace LeftMerge66546
def owner : Owner := ⟨.program ⟨214⟩, ⟨29591⟩⟩
def mergeEvent : Nat := 66546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }
def rhsRaw : List Term := Proof.Events258.exact66257RawTerms
def group : MergeGroup := .relation 66545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66545) (rhsResult := 66257)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29589⟩⟩) ⟨24663⟩ 66257) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66546

namespace LeftMerge66560
def owner : Owner := ⟨.program ⟨214⟩, ⟨22551⟩⟩
def mergeEvent : Nat := 66560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events259.exact66554RawTerms
def group : MergeGroup := .operator 65387 66554
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 66554) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22548⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66560

namespace LeftMerge66681
def owner : Owner := ⟨.program ⟨214⟩, ⟨16825⟩⟩
def mergeEvent : Nat := 66681
def frameStart : Nat := 66615
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events260.exact66677RawTerms
def rightRaw : List Term := Proof.Events260.exact66675RawTerms
def group : MergeGroup := .operator 66677 66675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66677) (leftOrdinal := 0)
    (rightResult := 66675) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66681

namespace LeftMerge66693
def owner : Owner := ⟨.program ⟨214⟩, ⟨29590⟩⟩
def mergeEvent : Nat := 66693
def frameStart : Nat := 66615
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }
def leftRaw : List Term := Proof.Events260.exact66689RawTerms
def rightRaw : List Term := Proof.Events260.exact66666RawTerms
def group : MergeGroup := .operator 66689 66666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66689) (leftOrdinal := 0)
    (rightResult := 66666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29589⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66693

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
