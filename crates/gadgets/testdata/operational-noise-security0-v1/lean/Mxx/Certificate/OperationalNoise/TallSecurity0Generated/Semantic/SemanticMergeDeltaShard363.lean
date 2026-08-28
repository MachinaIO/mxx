import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge60805
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60805
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17088⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60616RawTerms
def group : MergeGroup := .relation 60804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60804) (rhsResult := 60616)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17088⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 60616) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17088⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60805

namespace LeftMerge60806
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60806
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16801⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 28)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16801⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16801⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60806

namespace LeftMerge60808
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60808
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16801⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60616RawTerms
def group : MergeGroup := .relation 60807
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60807) (rhsResult := 60616)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16801⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 60616) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16801⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60808

namespace LeftMerge60809
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60809
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16682⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 27)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16682⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60809

namespace LeftMerge60811
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60811
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16682⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60616RawTerms
def group : MergeGroup := .relation 60810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60810) (rhsResult := 60616)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 60616) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16682⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60811

namespace LeftMerge60812
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60812
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18208⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 34)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18208⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60812

namespace LeftMerge60814
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60814
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18208⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60616RawTerms
def group : MergeGroup := .relation 60813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60813) (rhsResult := 60616)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 60616) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18208⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60814

namespace LeftMerge60815
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60815
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17907⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 32)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17907⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60815

namespace LeftMerge60817
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60817
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17907⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60616RawTerms
def group : MergeGroup := .relation 60816
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60816) (rhsResult := 60616)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 60616) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17907⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60817

namespace LeftMerge60818
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60818
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17123⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 30)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17123⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60818

namespace LeftMerge60820
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60820
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17123⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60616RawTerms
def group : MergeGroup := .relation 60819
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60819) (rhsResult := 60616)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 60616) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17123⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60820

namespace LeftMerge60821
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60821
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 26)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60821

namespace LeftMerge60823
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60823
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16311⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60616RawTerms
def group : MergeGroup := .relation 60822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60822) (rhsResult := 60616)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 60616) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16311⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60823

namespace LeftMerge60824
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60824
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 35)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60824

namespace LeftMerge60826
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60826
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18353⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60616RawTerms
def group : MergeGroup := .relation 60825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60825) (rhsResult := 60616)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18684⟩⟩) ⟨18620⟩ 60616) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18620⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18353⟩⟩], [⟨.program ⟨214⟩, ⟨18620⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60826

namespace LeftMerge60827
def owner : Owner := ⟨.program ⟨214⟩, ⟨18685⟩⟩
def mergeEvent : Nat := 60827
def frameStart : Nat := 60103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16108⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩] } }
def leftRaw : List Term := Proof.Events237.exact60778RawTerms
def rightRaw : List Term := Proof.Events236.exact60619RawTerms
def group : MergeGroup := .operator 60778 60619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60778) (leftOrdinal := 25)
    (rightResult := 60619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16108⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16108⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60827

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
