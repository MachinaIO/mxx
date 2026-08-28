import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge22452
def owner : Owner := ⟨.program ⟨214⟩, ⟨10154⟩⟩
def mergeEvent : Nat := 22452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events087.exact22443RawTerms
def rightRaw : List Term := Proof.Events029.exact7504RawTerms
def group : MergeGroup := .operator 22443 7504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22443) (leftOrdinal := 0)
    (rightResult := 7504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22452

namespace LeftMerge22457
def owner : Owner := ⟨.program ⟨214⟩, ⟨12989⟩⟩
def mergeEvent : Nat := 22457
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }
def leftRaw : List Term := Proof.Events087.exact22453RawTerms
def rightRaw : List Term := Proof.Events087.exact22423RawTerms
def group : MergeGroup := .operator 22453 22423
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22453) (leftOrdinal := 1)
    (rightResult := 22423) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6788⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩], [⟨.program ⟨214⟩, ⟨6788⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22457

namespace LeftMerge22465
def owner : Owner := ⟨.program ⟨214⟩, ⟨25620⟩⟩
def mergeEvent : Nat := 22465
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩] } }
def leftRaw : List Term := Proof.Events087.exact22459RawTerms
def rightRaw : List Term := Proof.Events087.exact22395RawTerms
def group : MergeGroup := .operator 22459 22395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22459) (leftOrdinal := 1)
    (rightResult := 22395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25619⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22465

namespace LeftMerge22467
def owner : Owner := ⟨.program ⟨214⟩, ⟨25620⟩⟩
def mergeEvent : Nat := 22467
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23338⟩⟩] } }
def rhsRaw : List Term := Proof.Events087.exact22392RawTerms
def group : MergeGroup := .relation 22466
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22466) (rhsResult := 22392)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25619⟩⟩) ⟨23338⟩ 22392) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23338⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], [⟨.program ⟨214⟩, ⟨23338⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22467

namespace LeftMerge22468
def owner : Owner := ⟨.program ⟨214⟩, ⟨25620⟩⟩
def mergeEvent : Nat := 22468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩] } }
def leftRaw : List Term := Proof.Events087.exact22459RawTerms
def rightRaw : List Term := Proof.Events087.exact22395RawTerms
def group : MergeGroup := .operator 22459 22395
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22459) (leftOrdinal := 0)
    (rightResult := 22395) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25619⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22468

namespace LeftMerge22482
def owner : Owner := ⟨.program ⟨214⟩, ⟨20119⟩⟩
def mergeEvent : Nat := 22482
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20116⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events087.exact22476RawTerms
def group : MergeGroup := .operator 21512 22476
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 22476) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20116⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20116⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22482

namespace LeftMerge22561
def owner : Owner := ⟨.program ⟨214⟩, ⟨12983⟩⟩
def mergeEvent : Nat := 22561
def frameStart : Nat := 22531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events088.exact22557RawTerms
def rightRaw : List Term := Proof.Events088.exact22554RawTerms
def group : MergeGroup := .operator 22557 22554
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22557) (leftOrdinal := 0)
    (rightResult := 22554) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10150⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22561

namespace LeftMerge22591
def owner : Owner := ⟨.program ⟨214⟩, ⟨13068⟩⟩
def mergeEvent : Nat := 22591
def frameStart : Nat := 22531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events088.exact22587RawTerms
def rightRaw : List Term := Proof.Events088.exact22585RawTerms
def group : MergeGroup := .operator 22587 22585
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22587) (leftOrdinal := 0)
    (rightResult := 22585) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22591

namespace LeftMerge22614
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def mergeEvent : Nat := 22614
def frameStart : Nat := 22531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events088.exact22610RawTerms
def rightRaw : List Term := Proof.Events088.exact22607RawTerms
def group : MergeGroup := .operator 22610 22607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22610) (leftOrdinal := 0)
    (rightResult := 22607) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22614

namespace LeftMerge22623
def owner : Owner := ⟨.program ⟨214⟩, ⟨25622⟩⟩
def mergeEvent : Nat := 22623
def frameStart : Nat := 22531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩] } }
def leftRaw : List Term := Proof.Events088.exact22619RawTerms
def rightRaw : List Term := Proof.Events088.exact22576RawTerms
def group : MergeGroup := .operator 22619 22576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22619) (leftOrdinal := 0)
    (rightResult := 22576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25619⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22623

namespace LeftMerge22624
def owner : Owner := ⟨.program ⟨214⟩, ⟨25622⟩⟩
def mergeEvent : Nat := 22624
def frameStart : Nat := 22531
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩] } }
def leftRaw : List Term := Proof.Events088.exact22619RawTerms
def rightRaw : List Term := Proof.Events088.exact22576RawTerms
def group : MergeGroup := .operator 22619 22576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22619) (leftOrdinal := 1)
    (rightResult := 22576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25619⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22624

namespace LeftMerge22626
def owner : Owner := ⟨.program ⟨214⟩, ⟨25622⟩⟩
def mergeEvent : Nat := 22626
def frameStart : Nat := 22531
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23338⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22573RawTerms
def group : MergeGroup := .relation 22625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22625) (rhsResult := 22573)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25619⟩⟩) ⟨23338⟩ 22573) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23338⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], [⟨.program ⟨214⟩, ⟨23338⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22626

namespace LeftMerge22634
def owner : Owner := ⟨.program ⟨214⟩, ⟨16766⟩⟩
def mergeEvent : Nat := 22634
def frameStart : Nat := 22531
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16764⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events088.exact22587RawTerms
def rightRaw : List Term := Proof.Events088.exact22630RawTerms
def group : MergeGroup := .operator 22587 22630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22587) (leftOrdinal := 0)
    (rightResult := 22630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16764⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16764⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22634

namespace LeftMerge22651
def owner : Owner := ⟨.program ⟨214⟩, ⟨20119⟩⟩
def mergeEvent : Nat := 22651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22648RawTerms
def group : MergeGroup := .relation 22650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22650) (rhsResult := 22648)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20116⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22649 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20116⟩⟩]⟩) (none) 22648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22651

namespace LeftMerge22652
def owner : Owner := ⟨.program ⟨214⟩, ⟨20119⟩⟩
def mergeEvent : Nat := 22652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22648RawTerms
def group : MergeGroup := .relation 22650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22650) (rhsResult := 22648)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20116⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22649 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20116⟩⟩]⟩) (none) 22648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25619⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22652

namespace LeftMerge22653
def owner : Owner := ⟨.program ⟨214⟩, ⟨20119⟩⟩
def mergeEvent : Nat := 22653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23338⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22648RawTerms
def group : MergeGroup := .relation 22650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22650) (rhsResult := 22648)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20116⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22649 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20116⟩⟩]⟩) (none) 22648) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23338⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10150⟩⟩, ⟨.program ⟨214⟩, ⟨12982⟩⟩], [⟨.program ⟨214⟩, ⟨23338⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22653

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
