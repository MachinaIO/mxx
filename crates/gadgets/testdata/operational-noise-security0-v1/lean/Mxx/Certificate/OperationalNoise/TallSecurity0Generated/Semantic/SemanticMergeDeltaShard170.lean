import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge29417
def owner : Owner := ⟨.program ⟨214⟩, ⟨26605⟩⟩
def mergeEvent : Nat := 29417
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩] } }
def leftRaw : List Term := Proof.Events114.exact29410RawTerms
def rightRaw : List Term := Proof.Events113.exact29133RawTerms
def group : MergeGroup := .operator 29410 29133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29410) (leftOrdinal := 1)
    (rightResult := 29133) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26603⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29417

namespace LeftMerge29419
def owner : Owner := ⟨.program ⟨214⟩, ⟨26605⟩⟩
def mergeEvent : Nat := 29419
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23793⟩⟩] } }
def rhsRaw : List Term := Proof.Events113.exact29130RawTerms
def group : MergeGroup := .relation 29418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29418) (rhsResult := 29130)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26603⟩⟩) ⟨23793⟩ 29130) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23793⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨23793⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29419

namespace LeftMerge29433
def owner : Owner := ⟨.program ⟨214⟩, ⟨20551⟩⟩
def mergeEvent : Nat := 29433
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events114.exact29427RawTerms
def group : MergeGroup := .operator 21512 29427
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 29427) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20548⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29433

namespace LeftMerge29554
def owner : Owner := ⟨.program ⟨214⟩, ⟨15007⟩⟩
def mergeEvent : Nat := 29554
def frameStart : Nat := 29488
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29550RawTerms
def rightRaw : List Term := Proof.Events115.exact29548RawTerms
def group : MergeGroup := .operator 29550 29548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29550) (leftOrdinal := 0)
    (rightResult := 29548) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29554

namespace LeftMerge29566
def owner : Owner := ⟨.program ⟨214⟩, ⟨26604⟩⟩
def mergeEvent : Nat := 29566
def frameStart : Nat := 29488
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29562RawTerms
def rightRaw : List Term := Proof.Events115.exact29539RawTerms
def group : MergeGroup := .operator 29562 29539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29562) (leftOrdinal := 0)
    (rightResult := 29539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26603⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29566

namespace LeftMerge29567
def owner : Owner := ⟨.program ⟨214⟩, ⟨26604⟩⟩
def mergeEvent : Nat := 29567
def frameStart : Nat := 29488
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29562RawTerms
def rightRaw : List Term := Proof.Events115.exact29539RawTerms
def group : MergeGroup := .operator 29562 29539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29562) (leftOrdinal := 1)
    (rightResult := 29539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26603⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29567

namespace LeftMerge29569
def owner : Owner := ⟨.program ⟨214⟩, ⟨26604⟩⟩
def mergeEvent : Nat := 29569
def frameStart : Nat := 29488
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23793⟩⟩] } }
def rhsRaw : List Term := Proof.Events115.exact29536RawTerms
def group : MergeGroup := .relation 29568
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29568) (rhsResult := 29536)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26603⟩⟩) ⟨23793⟩ 29536) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23793⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨23793⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29569

namespace LeftMerge29577
def owner : Owner := ⟨.program ⟨214⟩, ⟨15324⟩⟩
def mergeEvent : Nat := 29577
def frameStart : Nat := 29488
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29550RawTerms
def rightRaw : List Term := Proof.Events115.exact29573RawTerms
def group : MergeGroup := .operator 29550 29573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29550) (leftOrdinal := 0)
    (rightResult := 29573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29577

namespace LeftMerge29594
def owner : Owner := ⟨.program ⟨214⟩, ⟨20551⟩⟩
def mergeEvent : Nat := 29594
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }
def rhsRaw : List Term := Proof.Events115.exact29591RawTerms
def group : MergeGroup := .relation 29593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29593) (rhsResult := 29591)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29592 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩) (none) 29591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29594

namespace LeftMerge29595
def owner : Owner := ⟨.program ⟨214⟩, ⟨20551⟩⟩
def mergeEvent : Nat := 29595
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩] } }
def rhsRaw : List Term := Proof.Events115.exact29591RawTerms
def group : MergeGroup := .relation 29593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29593) (rhsResult := 29591)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29592 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩) (none) 29591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29595

namespace LeftMerge29596
def owner : Owner := ⟨.program ⟨214⟩, ⟨20551⟩⟩
def mergeEvent : Nat := 29596
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23793⟩⟩] } }
def rhsRaw : List Term := Proof.Events115.exact29591RawTerms
def group : MergeGroup := .relation 29593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29593) (rhsResult := 29591)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29592 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩) (none) 29591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23793⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨23793⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29596

namespace LeftMerge29597
def owner : Owner := ⟨.program ⟨214⟩, ⟨20551⟩⟩
def mergeEvent : Nat := 29597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events115.exact29591RawTerms
def group : MergeGroup := .relation 29593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29593) (rhsResult := 29591)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 29592 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩) (none) 29591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15322⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨15322⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29597

namespace LeftMerge29602
def owner : Owner := ⟨.program ⟨214⟩, ⟨26606⟩⟩
def mergeEvent : Nat := 29602
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29598RawTerms
def rightRaw : List Term := Proof.Events114.exact29420RawTerms
def group : MergeGroup := .operator 29598 29420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29598) (leftOrdinal := 0)
    (rightResult := 29420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26603⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29602

namespace LeftMerge29603
def owner : Owner := ⟨.program ⟨214⟩, ⟨26606⟩⟩
def mergeEvent : Nat := 29603
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23793⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29598RawTerms
def rightRaw : List Term := Proof.Events114.exact29420RawTerms
def group : MergeGroup := .operator 29598 29420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29598) (leftOrdinal := 2)
    (rightResult := 29420) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23793⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23793⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨14965⟩⟩], [⟨.program ⟨214⟩, ⟨23793⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29603

namespace LeftMerge29629
def owner : Owner := ⟨.program ⟨214⟩, ⟨10507⟩⟩
def mergeEvent : Nat := 29629
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events004.exact1233RawTerms
def rightRaw : List Term := Proof.Events083.exact21420RawTerms
def group : MergeGroup := .operator 1233 21420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1233) (leftOrdinal := 0)
    (rightResult := 21420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29629

namespace LeftMerge29634
def owner : Owner := ⟨.program ⟨214⟩, ⟨7342⟩⟩
def mergeEvent : Nat := 29634
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21290RawTerms
def rightRaw : List Term := Proof.Events058.exact14989RawTerms
def group : MergeGroup := .operator 21290 14989
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21290) (leftOrdinal := 0)
    (rightResult := 14989) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29634

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
