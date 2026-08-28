import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge36651
def owner : Owner := ⟨.program ⟨257⟩, ⟨65153⟩⟩
def mergeEvent : Nat := 36651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36644RawTerms
def rightRaw : List Term := Proof.Events142.exact36367RawTerms
def group : MergeGroup := .operator 36644 36367
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36644) (leftOrdinal := 1)
    (rightResult := 36367) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36651

namespace LeftMerge36653
def owner : Owner := ⟨.program ⟨257⟩, ⟨65153⟩⟩
def mergeEvent : Nat := 36653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64162⟩⟩] } }
def rhsRaw : List Term := Proof.Events142.exact36364RawTerms
def group : MergeGroup := .relation 36652
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36652) (rhsResult := 36364)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨65151⟩⟩) ⟨64162⟩ 36364) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64162⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨64162⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36653

namespace LeftMerge36667
def owner : Owner := ⟨.program ⟨257⟩, ⟨63859⟩⟩
def mergeEvent : Nat := 36667
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events143.exact36661RawTerms
def group : MergeGroup := .operator 32120 36661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 36661) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63856⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36667

namespace LeftMerge36788
def owner : Owner := ⟨.program ⟨257⟩, ⟨64324⟩⟩
def mergeEvent : Nat := 36788
def frameStart : Nat := 36722
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36784RawTerms
def rightRaw : List Term := Proof.Events143.exact36782RawTerms
def group : MergeGroup := .operator 36784 36782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36784) (leftOrdinal := 0)
    (rightResult := 36782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36788

namespace LeftMerge36800
def owner : Owner := ⟨.program ⟨257⟩, ⟨65152⟩⟩
def mergeEvent : Nat := 36800
def frameStart : Nat := 36722
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36796RawTerms
def rightRaw : List Term := Proof.Events143.exact36773RawTerms
def group : MergeGroup := .operator 36796 36773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36796) (leftOrdinal := 0)
    (rightResult := 36773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65151⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36800

namespace LeftMerge36801
def owner : Owner := ⟨.program ⟨257⟩, ⟨65152⟩⟩
def mergeEvent : Nat := 36801
def frameStart : Nat := 36722
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36796RawTerms
def rightRaw : List Term := Proof.Events143.exact36773RawTerms
def group : MergeGroup := .operator 36796 36773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36796) (leftOrdinal := 1)
    (rightResult := 36773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36801

namespace LeftMerge36803
def owner : Owner := ⟨.program ⟨257⟩, ⟨65152⟩⟩
def mergeEvent : Nat := 36803
def frameStart : Nat := 36722
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64162⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36770RawTerms
def group : MergeGroup := .relation 36802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36802) (rhsResult := 36770)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨65151⟩⟩) ⟨64162⟩ 36770) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64162⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨64162⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36803

namespace LeftMerge36811
def owner : Owner := ⟨.program ⟨257⟩, ⟨63254⟩⟩
def mergeEvent : Nat := 36811
def frameStart : Nat := 36722
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36784RawTerms
def rightRaw : List Term := Proof.Events143.exact36807RawTerms
def group : MergeGroup := .operator 36784 36807
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36784) (leftOrdinal := 0)
    (rightResult := 36807) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63252⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36811

namespace LeftMerge36828
def owner : Owner := ⟨.program ⟨257⟩, ⟨63859⟩⟩
def mergeEvent : Nat := 36828
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36825RawTerms
def group : MergeGroup := .relation 36827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36827) (rhsResult := 36825)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36826 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩]⟩) (none) 36825) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36828

namespace LeftMerge36829
def owner : Owner := ⟨.program ⟨257⟩, ⟨63859⟩⟩
def mergeEvent : Nat := 36829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36825RawTerms
def group : MergeGroup := .relation 36827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36827) (rhsResult := 36825)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36826 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩]⟩) (none) 36825) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36829

namespace LeftMerge36830
def owner : Owner := ⟨.program ⟨257⟩, ⟨63859⟩⟩
def mergeEvent : Nat := 36830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64162⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36825RawTerms
def group : MergeGroup := .relation 36827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36827) (rhsResult := 36825)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36826 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩]⟩) (none) 36825) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64162⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨64162⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36830

namespace LeftMerge36831
def owner : Owner := ⟨.program ⟨257⟩, ⟨63859⟩⟩
def mergeEvent : Nat := 36831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36825RawTerms
def group : MergeGroup := .relation 36827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36827) (rhsResult := 36825)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36826 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63856⟩⟩]⟩) (none) 36825) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63252⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨63252⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36831

namespace LeftMerge36836
def owner : Owner := ⟨.program ⟨257⟩, ⟨65154⟩⟩
def mergeEvent : Nat := 36836
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36832RawTerms
def rightRaw : List Term := Proof.Events143.exact36654RawTerms
def group : MergeGroup := .operator 36832 36654
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36832) (leftOrdinal := 0)
    (rightResult := 36654) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36836

namespace LeftMerge36837
def owner : Owner := ⟨.program ⟨257⟩, ⟨65154⟩⟩
def mergeEvent : Nat := 36837
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64162⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36832RawTerms
def rightRaw : List Term := Proof.Events143.exact36654RawTerms
def group : MergeGroup := .operator 36832 36654
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36832) (leftOrdinal := 2)
    (rightResult := 36654) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64162⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64162⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨64162⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36837

namespace LeftMerge36863
def owner : Owner := ⟨.program ⟨257⟩, ⟨25359⟩⟩
def mergeEvent : Nat := 36863
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events004.exact1072RawTerms
def rightRaw : List Term := Proof.Events125.exact32028RawTerms
def group : MergeGroup := .operator 1072 32028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1072) (leftOrdinal := 0)
    (rightResult := 32028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25358⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36863

namespace LeftMerge36868
def owner : Owner := ⟨.program ⟨257⟩, ⟨11607⟩⟩
def mergeEvent : Nat := 36868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events124.exact31898RawTerms
def rightRaw : List Term := Proof.Events086.exact22090RawTerms
def group : MergeGroup := .operator 31898 22090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31898) (leftOrdinal := 0)
    (rightResult := 22090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36868

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
