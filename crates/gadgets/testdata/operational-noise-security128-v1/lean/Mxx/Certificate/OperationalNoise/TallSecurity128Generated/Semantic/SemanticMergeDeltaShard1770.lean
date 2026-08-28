import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge286650
def owner : Owner := ⟨.program ⟨257⟩, ⟨55436⟩⟩
def mergeEvent : Nat := 286650
def frameStart : Nat := 286560
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩] } }
def leftRaw : List Term := Proof.Events1119.exact286646RawTerms
def rightRaw : List Term := Proof.Events1119.exact286605RawTerms
def group : MergeGroup := .operator 286646 286605
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286646) (leftOrdinal := 0)
    (rightResult := 286605) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55433⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286650

namespace LeftMerge286651
def owner : Owner := ⟨.program ⟨257⟩, ⟨55436⟩⟩
def mergeEvent : Nat := 286651
def frameStart : Nat := 286560
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩] } }
def leftRaw : List Term := Proof.Events1119.exact286646RawTerms
def rightRaw : List Term := Proof.Events1119.exact286605RawTerms
def group : MergeGroup := .operator 286646 286605
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286646) (leftOrdinal := 1)
    (rightResult := 286605) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55433⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286651

namespace LeftMerge286653
def owner : Owner := ⟨.program ⟨257⟩, ⟨55436⟩⟩
def mergeEvent : Nat := 286653
def frameStart : Nat := 286560
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54953⟩⟩] } }
def rhsRaw : List Term := Proof.Events1119.exact286602RawTerms
def group : MergeGroup := .relation 286652
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286652) (rhsResult := 286602)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55433⟩⟩) ⟨54953⟩ 286602) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54953⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], [⟨.program ⟨257⟩, ⟨54953⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286653

namespace LeftMerge286661
def owner : Owner := ⟨.program ⟨257⟩, ⟨53822⟩⟩
def mergeEvent : Nat := 286661
def frameStart : Nat := 286560
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1119.exact286616RawTerms
def rightRaw : List Term := Proof.Events1119.exact286657RawTerms
def group : MergeGroup := .operator 286616 286657
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286616) (leftOrdinal := 0)
    (rightResult := 286657) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286661

namespace LeftMerge286678
def owner : Owner := ⟨.program ⟨257⟩, ⟨54372⟩⟩
def mergeEvent : Nat := 286678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events1119.exact286675RawTerms
def group : MergeGroup := .relation 286677
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286677) (rhsResult := 286675)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286676 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54369⟩⟩]⟩) (none) 286675) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286678

namespace LeftMerge286679
def owner : Owner := ⟨.program ⟨257⟩, ⟨54372⟩⟩
def mergeEvent : Nat := 286679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩] } }
def rhsRaw : List Term := Proof.Events1119.exact286675RawTerms
def group : MergeGroup := .relation 286677
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286677) (rhsResult := 286675)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286676 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54369⟩⟩]⟩) (none) 286675) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286679

namespace LeftMerge286680
def owner : Owner := ⟨.program ⟨257⟩, ⟨54372⟩⟩
def mergeEvent : Nat := 286680
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54953⟩⟩] } }
def rhsRaw : List Term := Proof.Events1119.exact286675RawTerms
def group : MergeGroup := .relation 286677
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286677) (rhsResult := 286675)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286676 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54369⟩⟩]⟩) (none) 286675) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54953⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], [⟨.program ⟨257⟩, ⟨54953⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286680

namespace LeftMerge286681
def owner : Owner := ⟨.program ⟨257⟩, ⟨54372⟩⟩
def mergeEvent : Nat := 286681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1119.exact286675RawTerms
def group : MergeGroup := .relation 286677
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286677) (rhsResult := 286675)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54369⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 286676 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54369⟩⟩]⟩) (none) 286675) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286681

namespace LeftMerge286686
def owner : Owner := ⟨.program ⟨257⟩, ⟨55435⟩⟩
def mergeEvent : Nat := 286686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54953⟩⟩] } }
def leftRaw : List Term := Proof.Events1119.exact286682RawTerms
def rightRaw : List Term := Proof.Events1119.exact286498RawTerms
def group : MergeGroup := .operator 286682 286498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286682) (leftOrdinal := 2)
    (rightResult := 286498) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54953⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54953⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨24698⟩⟩, ⟨.program ⟨257⟩, ⟨53363⟩⟩], [⟨.program ⟨257⟩, ⟨54953⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286686

namespace LeftMerge286687
def owner : Owner := ⟨.program ⟨257⟩, ⟨55435⟩⟩
def mergeEvent : Nat := 286687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩] } }
def leftRaw : List Term := Proof.Events1119.exact286682RawTerms
def rightRaw : List Term := Proof.Events1119.exact286498RawTerms
def group : MergeGroup := .operator 286682 286498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286682) (leftOrdinal := 1)
    (rightResult := 286498) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55433⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286687

namespace LeftMerge286695
def owner : Owner := ⟨.program ⟨257⟩, ⟨55748⟩⟩
def mergeEvent : Nat := 286695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩] } }
def leftRaw : List Term := Proof.Events1119.exact286689RawTerms
def rightRaw : List Term := Proof.Events1118.exact286414RawTerms
def group : MergeGroup := .operator 286689 286414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286689) (leftOrdinal := 0)
    (rightResult := 286414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55746⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286695

namespace LeftMerge286696
def owner : Owner := ⟨.program ⟨257⟩, ⟨55748⟩⟩
def mergeEvent : Nat := 286696
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩] } }
def leftRaw : List Term := Proof.Events1119.exact286689RawTerms
def rightRaw : List Term := Proof.Events1118.exact286414RawTerms
def group : MergeGroup := .operator 286689 286414
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286689) (leftOrdinal := 1)
    (rightResult := 286414) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55746⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286696

namespace LeftMerge286698
def owner : Owner := ⟨.program ⟨257⟩, ⟨55748⟩⟩
def mergeEvent : Nat := 286698
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55087⟩⟩] } }
def rhsRaw : List Term := Proof.Events1118.exact286411RawTerms
def group : MergeGroup := .relation 286697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 286697) (rhsResult := 286411)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55746⟩⟩) ⟨55087⟩ 286411) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55087⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩, ⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨55087⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge286698

namespace LeftMerge286712
def owner : Owner := ⟨.program ⟨257⟩, ⟨54619⟩⟩
def mergeEvent : Nat := 286712
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54616⟩⟩] } }
def leftRaw : List Term := Proof.Events1096.exact280745RawTerms
def rightRaw : List Term := Proof.Events1119.exact286706RawTerms
def group : MergeGroup := .operator 280745 286706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 280745) (leftOrdinal := 0)
    (rightResult := 286706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54616⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2378⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286712

namespace LeftMerge286833
def owner : Owner := ⟨.program ⟨257⟩, ⟨55324⟩⟩
def mergeEvent : Nat := 286833
def frameStart : Nat := 286767
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1120.exact286829RawTerms
def rightRaw : List Term := Proof.Events1120.exact286827RawTerms
def group : MergeGroup := .operator 286829 286827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286829) (leftOrdinal := 0)
    (rightResult := 286827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286833

namespace LeftMerge286845
def owner : Owner := ⟨.program ⟨257⟩, ⟨55747⟩⟩
def mergeEvent : Nat := 286845
def frameStart : Nat := 286767
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩] } }
def leftRaw : List Term := Proof.Events1120.exact286841RawTerms
def rightRaw : List Term := Proof.Events1120.exact286818RawTerms
def group : MergeGroup := .operator 286841 286818
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 286841) (leftOrdinal := 0)
    (rightResult := 286818) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55746⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55746⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge286845

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
