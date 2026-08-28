import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge38553
def owner : Owner := ⟨.program ⟨214⟩, ⟨19899⟩⟩
def mergeEvent : Nat := 38553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events150.exact38547RawTerms
def group : MergeGroup := .operator 36137 38547
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 38547) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19896⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38553

namespace LeftMerge38632
def owner : Owner := ⟨.program ⟨214⟩, ⟨12387⟩⟩
def mergeEvent : Nat := 38632
def frameStart : Nat := 38602
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events150.exact38628RawTerms
def rightRaw : List Term := Proof.Events150.exact38625RawTerms
def group : MergeGroup := .operator 38628 38625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38628) (leftOrdinal := 0)
    (rightResult := 38625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9830⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38632

namespace LeftMerge38662
def owner : Owner := ⟨.program ⟨214⟩, ⟨12476⟩⟩
def mergeEvent : Nat := 38662
def frameStart : Nat := 38602
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38658RawTerms
def rightRaw : List Term := Proof.Events151.exact38656RawTerms
def group : MergeGroup := .operator 38658 38656
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38658) (leftOrdinal := 0)
    (rightResult := 38656) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38662

namespace LeftMerge38685
def owner : Owner := ⟨.program ⟨214⟩, ⟨7869⟩⟩
def mergeEvent : Nat := 38685
def frameStart : Nat := 38602
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38681RawTerms
def rightRaw : List Term := Proof.Events151.exact38678RawTerms
def group : MergeGroup := .operator 38681 38678
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38681) (leftOrdinal := 0)
    (rightResult := 38678) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7867⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38685

namespace LeftMerge38694
def owner : Owner := ⟨.program ⟨214⟩, ⟨25386⟩⟩
def mergeEvent : Nat := 38694
def frameStart : Nat := 38602
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38690RawTerms
def rightRaw : List Term := Proof.Events150.exact38647RawTerms
def group : MergeGroup := .operator 38690 38647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38690) (leftOrdinal := 0)
    (rightResult := 38647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25383⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38694

namespace LeftMerge38695
def owner : Owner := ⟨.program ⟨214⟩, ⟨25386⟩⟩
def mergeEvent : Nat := 38695
def frameStart : Nat := 38602
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38690RawTerms
def rightRaw : List Term := Proof.Events150.exact38647RawTerms
def group : MergeGroup := .operator 38690 38647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38690) (leftOrdinal := 1)
    (rightResult := 38647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25383⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38695

namespace LeftMerge38697
def owner : Owner := ⟨.program ⟨214⟩, ⟨25386⟩⟩
def mergeEvent : Nat := 38697
def frameStart : Nat := 38602
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23210⟩⟩] } }
def rhsRaw : List Term := Proof.Events150.exact38644RawTerms
def group : MergeGroup := .relation 38696
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38696) (rhsResult := 38644)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25383⟩⟩) ⟨23210⟩ 38644) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23210⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], [⟨.program ⟨214⟩, ⟨23210⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38697

namespace LeftMerge38705
def owner : Owner := ⟨.program ⟨214⟩, ⟨16475⟩⟩
def mergeEvent : Nat := 38705
def frameStart : Nat := 38602
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16473⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38658RawTerms
def rightRaw : List Term := Proof.Events151.exact38701RawTerms
def group : MergeGroup := .operator 38658 38701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38658) (leftOrdinal := 0)
    (rightResult := 38701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16473⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38705

namespace LeftMerge38722
def owner : Owner := ⟨.program ⟨214⟩, ⟨19899⟩⟩
def mergeEvent : Nat := 38722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }
def rhsRaw : List Term := Proof.Events151.exact38719RawTerms
def group : MergeGroup := .relation 38721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38721) (rhsResult := 38719)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38720 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩]⟩) (none) 38719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38722

namespace LeftMerge38723
def owner : Owner := ⟨.program ⟨214⟩, ⟨19899⟩⟩
def mergeEvent : Nat := 38723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩] } }
def rhsRaw : List Term := Proof.Events151.exact38719RawTerms
def group : MergeGroup := .relation 38721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38721) (rhsResult := 38719)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38720 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩]⟩) (none) 38719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38723

namespace LeftMerge38724
def owner : Owner := ⟨.program ⟨214⟩, ⟨19899⟩⟩
def mergeEvent : Nat := 38724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23210⟩⟩] } }
def rhsRaw : List Term := Proof.Events151.exact38719RawTerms
def group : MergeGroup := .relation 38721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38721) (rhsResult := 38719)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38720 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩]⟩) (none) 38719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23210⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], [⟨.program ⟨214⟩, ⟨23210⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38724

namespace LeftMerge38725
def owner : Owner := ⟨.program ⟨214⟩, ⟨19899⟩⟩
def mergeEvent : Nat := 38725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events151.exact38719RawTerms
def group : MergeGroup := .relation 38721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 38721) (rhsResult := 38719)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 38720 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19896⟩⟩]⟩) (none) 38719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16473⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38725

namespace LeftMerge38730
def owner : Owner := ⟨.program ⟨214⟩, ⟨25385⟩⟩
def mergeEvent : Nat := 38730
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23210⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38726RawTerms
def rightRaw : List Term := Proof.Events150.exact38540RawTerms
def group : MergeGroup := .operator 38726 38540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38726) (leftOrdinal := 2)
    (rightResult := 38540) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23210⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23210⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨9830⟩⟩, ⟨.program ⟨214⟩, ⟨12386⟩⟩], [⟨.program ⟨214⟩, ⟨23210⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38730

namespace LeftMerge38731
def owner : Owner := ⟨.program ⟨214⟩, ⟨25385⟩⟩
def mergeEvent : Nat := 38731
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38726RawTerms
def rightRaw : List Term := Proof.Events150.exact38540RawTerms
def group : MergeGroup := .operator 38726 38540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38726) (leftOrdinal := 1)
    (rightResult := 38540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6765⟩⟩, ⟨.program ⟨214⟩, ⟨7867⟩⟩, ⟨.program ⟨214⟩, ⟨25383⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38731

namespace LeftMerge38739
def owner : Owner := ⟨.program ⟨214⟩, ⟨28979⟩⟩
def mergeEvent : Nat := 38739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28977⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38733RawTerms
def rightRaw : List Term := Proof.Events150.exact38456RawTerms
def group : MergeGroup := .operator 38733 38456
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38733) (leftOrdinal := 0)
    (rightResult := 38456) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28977⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28977⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge38739

namespace LeftMerge38740
def owner : Owner := ⟨.program ⟨214⟩, ⟨28979⟩⟩
def mergeEvent : Nat := 38740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28977⟩⟩] } }
def leftRaw : List Term := Proof.Events151.exact38733RawTerms
def rightRaw : List Term := Proof.Events150.exact38456RawTerms
def group : MergeGroup := .operator 38733 38456
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 38733) (leftOrdinal := 1)
    (rightResult := 38456) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28977⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16473⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28977⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge38740

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
