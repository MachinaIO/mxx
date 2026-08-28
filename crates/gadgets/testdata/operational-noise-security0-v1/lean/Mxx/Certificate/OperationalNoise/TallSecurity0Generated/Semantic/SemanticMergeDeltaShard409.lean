import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge66694
def owner : Owner := ⟨.program ⟨214⟩, ⟨29590⟩⟩
def mergeEvent : Nat := 66694
def frameStart : Nat := 66615
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }
def leftRaw : List Term := Proof.Events260.exact66689RawTerms
def rightRaw : List Term := Proof.Events260.exact66666RawTerms
def group : MergeGroup := .operator 66689 66666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66689) (leftOrdinal := 1)
    (rightResult := 66666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29589⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66694

namespace LeftMerge66696
def owner : Owner := ⟨.program ⟨214⟩, ⟨29590⟩⟩
def mergeEvent : Nat := 66696
def frameStart : Nat := 66615
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }
def rhsRaw : List Term := Proof.Events260.exact66663RawTerms
def group : MergeGroup := .relation 66695
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66695) (rhsResult := 66663)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29589⟩⟩) ⟨24663⟩ 66663) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66696

namespace LeftMerge66704
def owner : Owner := ⟨.program ⟨214⟩, ⟨16796⟩⟩
def mergeEvent : Nat := 66704
def frameStart : Nat := 66615
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events260.exact66677RawTerms
def rightRaw : List Term := Proof.Events260.exact66700RawTerms
def group : MergeGroup := .operator 66677 66700
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66677) (leftOrdinal := 0)
    (rightResult := 66700) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66704

namespace LeftMerge66721
def owner : Owner := ⟨.program ⟨214⟩, ⟨22551⟩⟩
def mergeEvent : Nat := 66721
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }
def rhsRaw : List Term := Proof.Events260.exact66718RawTerms
def group : MergeGroup := .relation 66720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66720) (rhsResult := 66718)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66719 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩) (none) 66718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66721

namespace LeftMerge66722
def owner : Owner := ⟨.program ⟨214⟩, ⟨22551⟩⟩
def mergeEvent : Nat := 66722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }
def rhsRaw : List Term := Proof.Events260.exact66718RawTerms
def group : MergeGroup := .relation 66720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66720) (rhsResult := 66718)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66719 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩) (none) 66718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66722

namespace LeftMerge66723
def owner : Owner := ⟨.program ⟨214⟩, ⟨22551⟩⟩
def mergeEvent : Nat := 66723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }
def rhsRaw : List Term := Proof.Events260.exact66718RawTerms
def group : MergeGroup := .relation 66720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66720) (rhsResult := 66718)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66719 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩) (none) 66718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66723

namespace LeftMerge66724
def owner : Owner := ⟨.program ⟨214⟩, ⟨22551⟩⟩
def mergeEvent : Nat := 66724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events260.exact66718RawTerms
def group : MergeGroup := .relation 66720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66720) (rhsResult := 66718)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66719 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩) (none) 66718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66724

namespace LeftMerge66729
def owner : Owner := ⟨.program ⟨214⟩, ⟨29592⟩⟩
def mergeEvent : Nat := 66729
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }
def leftRaw : List Term := Proof.Events260.exact66725RawTerms
def rightRaw : List Term := Proof.Events259.exact66547RawTerms
def group : MergeGroup := .operator 66725 66547
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66725) (leftOrdinal := 0)
    (rightResult := 66547) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66729

namespace LeftMerge66730
def owner : Owner := ⟨.program ⟨214⟩, ⟨29592⟩⟩
def mergeEvent : Nat := 66730
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }
def leftRaw : List Term := Proof.Events260.exact66725RawTerms
def rightRaw : List Term := Proof.Events259.exact66547RawTerms
def group : MergeGroup := .operator 66725 66547
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66725) (leftOrdinal := 2)
    (rightResult := 66547) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24663⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16748⟩⟩], [⟨.program ⟨214⟩, ⟨24663⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66730

namespace LeftMerge66756
def owner : Owner := ⟨.program ⟨214⟩, ⟨12757⟩⟩
def mergeEvent : Nat := 66756
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events012.exact3155RawTerms
def rightRaw : List Term := Proof.Events255.exact65295RawTerms
def group : MergeGroup := .operator 3155 65295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3155) (leftOrdinal := 0)
    (rightResult := 65295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12754⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66756

namespace LeftMerge66761
def owner : Owner := ⟨.program ⟨214⟩, ⟨7205⟩⟩
def mergeEvent : Nat := 66761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65165RawTerms
def rightRaw : List Term := Proof.Events031.exact7975RawTerms
def group : MergeGroup := .operator 65165 7975
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65165) (leftOrdinal := 0)
    (rightResult := 7975) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66761

namespace LeftMerge66778
def owner : Owner := ⟨.program ⟨214⟩, ⟨12760⟩⟩
def mergeEvent : Nat := 66778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events260.exact66772RawTerms
def rightRaw : List Term := Proof.Events012.exact3158RawTerms
def group : MergeGroup := .operator 66772 3158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66772) (leftOrdinal := 1)
    (rightResult := 3158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10025⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩, ⟨.program ⟨214⟩, ⟨12754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66778

namespace LeftMerge66779
def owner : Owner := ⟨.program ⟨214⟩, ⟨12760⟩⟩
def mergeEvent : Nat := 66779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }
def leftRaw : List Term := Proof.Events260.exact66772RawTerms
def rightRaw : List Term := Proof.Events012.exact3158RawTerms
def group : MergeGroup := .operator 66772 3158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66772) (leftOrdinal := 0)
    (rightResult := 3158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6787⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10025⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6787⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66779

namespace LeftMerge66784
def owner : Owner := ⟨.program ⟨214⟩, ⟨10026⟩⟩
def mergeEvent : Nat := 66784
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events012.exact3158RawTerms
def rightRaw : List Term := Proof.Events255.exact65295RawTerms
def group : MergeGroup := .operator 3158 65295
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3158) (leftOrdinal := 0)
    (rightResult := 65295) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10025⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66784

namespace LeftMerge66789
def owner : Owner := ⟨.program ⟨214⟩, ⟨7185⟩⟩
def mergeEvent : Nat := 66789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } }
def leftRaw : List Term := Proof.Events254.exact65165RawTerms
def rightRaw : List Term := Proof.Events031.exact8016RawTerms
def group : MergeGroup := .operator 65165 8016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65165) (leftOrdinal := 0)
    (rightResult := 8016) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6767⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6767⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66789

namespace LeftMerge66806
def owner : Owner := ⟨.program ⟨214⟩, ⟨10029⟩⟩
def mergeEvent : Nat := 66806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩] } }
def leftRaw : List Term := Proof.Events260.exact66800RawTerms
def rightRaw : List Term := Proof.Events031.exact8005RawTerms
def group : MergeGroup := .operator 66800 8005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66800) (leftOrdinal := 1)
    (rightResult := 8005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7873⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨10025⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66806

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
