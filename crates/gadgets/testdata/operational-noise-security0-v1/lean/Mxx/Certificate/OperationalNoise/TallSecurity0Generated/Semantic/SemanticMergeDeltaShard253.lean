import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge42409
def owner : Owner := ⟨.program ⟨214⟩, ⟨19323⟩⟩
def mergeEvent : Nat := 42409
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events165.exact42403RawTerms
def group : MergeGroup := .operator 36137 42403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 42403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19320⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42409

namespace LeftMerge42488
def owner : Owner := ⟨.program ⟨214⟩, ⟨13575⟩⟩
def mergeEvent : Nat := 42488
def frameStart : Nat := 42458
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events165.exact42484RawTerms
def rightRaw : List Term := Proof.Events165.exact42481RawTerms
def group : MergeGroup := .operator 42484 42481
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42484) (leftOrdinal := 0)
    (rightResult := 42481) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11225⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42488

namespace LeftMerge42518
def owner : Owner := ⟨.program ⟨214⟩, ⟨13673⟩⟩
def mergeEvent : Nat := 42518
def frameStart : Nat := 42458
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events166.exact42514RawTerms
def rightRaw : List Term := Proof.Events166.exact42512RawTerms
def group : MergeGroup := .operator 42514 42512
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42514) (leftOrdinal := 0)
    (rightResult := 42512) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42518

namespace LeftMerge42541
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def mergeEvent : Nat := 42541
def frameStart : Nat := 42458
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }
def leftRaw : List Term := Proof.Events166.exact42537RawTerms
def rightRaw : List Term := Proof.Events166.exact42534RawTerms
def group : MergeGroup := .operator 42537 42534
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42537) (leftOrdinal := 0)
    (rightResult := 42534) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7843⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42541

namespace LeftMerge42550
def owner : Owner := ⟨.program ⟨214⟩, ⟨25848⟩⟩
def mergeEvent : Nat := 42550
def frameStart : Nat := 42458
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩] } }
def leftRaw : List Term := Proof.Events166.exact42546RawTerms
def rightRaw : List Term := Proof.Events166.exact42503RawTerms
def group : MergeGroup := .operator 42546 42503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42546) (leftOrdinal := 0)
    (rightResult := 42503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25845⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42550

namespace LeftMerge42551
def owner : Owner := ⟨.program ⟨214⟩, ⟨25848⟩⟩
def mergeEvent : Nat := 42551
def frameStart : Nat := 42458
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩] } }
def leftRaw : List Term := Proof.Events166.exact42546RawTerms
def rightRaw : List Term := Proof.Events166.exact42503RawTerms
def group : MergeGroup := .operator 42546 42503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42546) (leftOrdinal := 1)
    (rightResult := 42503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25845⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42551

namespace LeftMerge42553
def owner : Owner := ⟨.program ⟨214⟩, ⟨25848⟩⟩
def mergeEvent : Nat := 42553
def frameStart : Nat := 42458
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23462⟩⟩] } }
def rhsRaw : List Term := Proof.Events166.exact42500RawTerms
def group : MergeGroup := .relation 42552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42552) (rhsResult := 42500)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25845⟩⟩) ⟨23462⟩ 42500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23462⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], [⟨.program ⟨214⟩, ⟨23462⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42553

namespace LeftMerge42561
def owner : Owner := ⟨.program ⟨214⟩, ⟨15593⟩⟩
def mergeEvent : Nat := 42561
def frameStart : Nat := 42458
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events166.exact42514RawTerms
def rightRaw : List Term := Proof.Events166.exact42557RawTerms
def group : MergeGroup := .operator 42514 42557
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42514) (leftOrdinal := 0)
    (rightResult := 42557) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15591⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42561

namespace LeftMerge42578
def owner : Owner := ⟨.program ⟨214⟩, ⟨19323⟩⟩
def mergeEvent : Nat := 42578
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩] } }
def rhsRaw : List Term := Proof.Events166.exact42575RawTerms
def group : MergeGroup := .relation 42577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42577) (rhsResult := 42575)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 42576 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩) (none) 42575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42578

namespace LeftMerge42579
def owner : Owner := ⟨.program ⟨214⟩, ⟨19323⟩⟩
def mergeEvent : Nat := 42579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩] } }
def rhsRaw : List Term := Proof.Events166.exact42575RawTerms
def group : MergeGroup := .relation 42577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42577) (rhsResult := 42575)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 42576 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩) (none) 42575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42579

namespace LeftMerge42580
def owner : Owner := ⟨.program ⟨214⟩, ⟨19323⟩⟩
def mergeEvent : Nat := 42580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23462⟩⟩] } }
def rhsRaw : List Term := Proof.Events166.exact42575RawTerms
def group : MergeGroup := .relation 42577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42577) (rhsResult := 42575)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 42576 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩) (none) 42575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23462⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], [⟨.program ⟨214⟩, ⟨23462⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42580

namespace LeftMerge42581
def owner : Owner := ⟨.program ⟨214⟩, ⟨19323⟩⟩
def mergeEvent : Nat := 42581
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events166.exact42575RawTerms
def group : MergeGroup := .relation 42577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 42577) (rhsResult := 42575)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 42576 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩) (none) 42575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42581

namespace LeftMerge42586
def owner : Owner := ⟨.program ⟨214⟩, ⟨25847⟩⟩
def mergeEvent : Nat := 42586
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23462⟩⟩] } }
def leftRaw : List Term := Proof.Events166.exact42582RawTerms
def rightRaw : List Term := Proof.Events165.exact42396RawTerms
def group : MergeGroup := .operator 42582 42396
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42582) (leftOrdinal := 2)
    (rightResult := 42396) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23462⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23462⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11225⟩⟩, ⟨.program ⟨214⟩, ⟨13574⟩⟩], [⟨.program ⟨214⟩, ⟨23462⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42586

namespace LeftMerge42587
def owner : Owner := ⟨.program ⟨214⟩, ⟨25847⟩⟩
def mergeEvent : Nat := 42587
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩] } }
def leftRaw : List Term := Proof.Events166.exact42582RawTerms
def rightRaw : List Term := Proof.Events165.exact42396RawTerms
def group : MergeGroup := .operator 42582 42396
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42582) (leftOrdinal := 1)
    (rightResult := 42396) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25845⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42587

namespace LeftMerge42595
def owner : Owner := ⟨.program ⟨214⟩, ⟨27243⟩⟩
def mergeEvent : Nat := 42595
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27241⟩⟩] } }
def leftRaw : List Term := Proof.Events166.exact42589RawTerms
def rightRaw : List Term := Proof.Events165.exact42312RawTerms
def group : MergeGroup := .operator 42589 42312
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42589) (leftOrdinal := 0)
    (rightResult := 42312) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6694⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27241⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6694⟩⟩, ⟨.program ⟨214⟩, ⟨27241⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge42595

namespace LeftMerge42596
def owner : Owner := ⟨.program ⟨214⟩, ⟨27243⟩⟩
def mergeEvent : Nat := 42596
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27241⟩⟩] } }
def leftRaw : List Term := Proof.Events166.exact42589RawTerms
def rightRaw : List Term := Proof.Events165.exact42312RawTerms
def group : MergeGroup := .operator 42589 42312
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42589) (leftOrdinal := 1)
    (rightResult := 42312) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27241⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15591⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27241⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge42596

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
