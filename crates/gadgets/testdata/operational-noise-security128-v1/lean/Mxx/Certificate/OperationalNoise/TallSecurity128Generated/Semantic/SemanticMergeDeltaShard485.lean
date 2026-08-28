import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge81956
def owner : Owner := ⟨.program ⟨257⟩, ⟨54492⟩⟩
def mergeEvent : Nat := 81956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55025⟩⟩] } }
def rhsRaw : List Term := Proof.Events320.exact81951RawTerms
def group : MergeGroup := .relation 81953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81953) (rhsResult := 81951)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81952 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54489⟩⟩]⟩) (none) 81951) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55025⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81956

namespace LeftMerge81957
def owner : Owner := ⟨.program ⟨257⟩, ⟨54492⟩⟩
def mergeEvent : Nat := 81957
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events320.exact81951RawTerms
def group : MergeGroup := .relation 81953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81953) (rhsResult := 81951)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54489⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 81952 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54489⟩⟩]⟩) (none) 81951) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81957

namespace LeftMerge81962
def owner : Owner := ⟨.program ⟨257⟩, ⟨55567⟩⟩
def mergeEvent : Nat := 81962
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55025⟩⟩] } }
def leftRaw : List Term := Proof.Events320.exact81958RawTerms
def rightRaw : List Term := Proof.Events319.exact81772RawTerms
def group : MergeGroup := .operator 81958 81772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81958) (leftOrdinal := 2)
    (rightResult := 81772) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55025⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55025⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨24842⟩⟩, ⟨.program ⟨257⟩, ⟨53687⟩⟩], [⟨.program ⟨257⟩, ⟨55025⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81962

namespace LeftMerge81963
def owner : Owner := ⟨.program ⟨257⟩, ⟨55567⟩⟩
def mergeEvent : Nat := 81963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩] } }
def leftRaw : List Term := Proof.Events320.exact81958RawTerms
def rightRaw : List Term := Proof.Events319.exact81772RawTerms
def group : MergeGroup := .operator 81958 81772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81958) (leftOrdinal := 1)
    (rightResult := 81772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81963

namespace LeftMerge81971
def owner : Owner := ⟨.program ⟨257⟩, ⟨56120⟩⟩
def mergeEvent : Nat := 81971
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩] } }
def leftRaw : List Term := Proof.Events320.exact81965RawTerms
def rightRaw : List Term := Proof.Events319.exact81688RawTerms
def group : MergeGroup := .operator 81965 81688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81965) (leftOrdinal := 0)
    (rightResult := 81688) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56118⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81971

namespace LeftMerge81972
def owner : Owner := ⟨.program ⟨257⟩, ⟨56120⟩⟩
def mergeEvent : Nat := 81972
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩] } }
def leftRaw : List Term := Proof.Events320.exact81965RawTerms
def rightRaw : List Term := Proof.Events319.exact81688RawTerms
def group : MergeGroup := .operator 81965 81688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 81965) (leftOrdinal := 1)
    (rightResult := 81688) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56118⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81972

namespace LeftMerge81974
def owner : Owner := ⟨.program ⟨257⟩, ⟨56120⟩⟩
def mergeEvent : Nat := 81974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55195⟩⟩] } }
def rhsRaw : List Term := Proof.Events319.exact81685RawTerms
def group : MergeGroup := .relation 81973
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 81973) (rhsResult := 81685)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56118⟩⟩) ⟨55195⟩ 81685) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge81974

namespace LeftMerge81988
def owner : Owner := ⟨.program ⟨257⟩, ⟨54859⟩⟩
def mergeEvent : Nat := 81988
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events320.exact81982RawTerms
def group : MergeGroup := .operator 75995 81982
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 81982) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54856⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge81988

namespace LeftMerge82109
def owner : Owner := ⟨.program ⟨257⟩, ⟨55372⟩⟩
def mergeEvent : Nat := 82109
def frameStart : Nat := 82043
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events320.exact82105RawTerms
def rightRaw : List Term := Proof.Events320.exact82103RawTerms
def group : MergeGroup := .operator 82105 82103
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82105) (leftOrdinal := 0)
    (rightResult := 82103) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82109

namespace LeftMerge82121
def owner : Owner := ⟨.program ⟨257⟩, ⟨56119⟩⟩
def mergeEvent : Nat := 82121
def frameStart : Nat := 82043
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩] } }
def leftRaw : List Term := Proof.Events320.exact82117RawTerms
def rightRaw : List Term := Proof.Events320.exact82094RawTerms
def group : MergeGroup := .operator 82117 82094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82117) (leftOrdinal := 0)
    (rightResult := 82094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56118⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82121

namespace LeftMerge82122
def owner : Owner := ⟨.program ⟨257⟩, ⟨56119⟩⟩
def mergeEvent : Nat := 82122
def frameStart : Nat := 82043
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩] } }
def leftRaw : List Term := Proof.Events320.exact82117RawTerms
def rightRaw : List Term := Proof.Events320.exact82094RawTerms
def group : MergeGroup := .operator 82117 82094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82117) (leftOrdinal := 1)
    (rightResult := 82094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56118⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82122

namespace LeftMerge82124
def owner : Owner := ⟨.program ⟨257⟩, ⟨56119⟩⟩
def mergeEvent : Nat := 82124
def frameStart : Nat := 82043
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55195⟩⟩] } }
def rhsRaw : List Term := Proof.Events320.exact82091RawTerms
def group : MergeGroup := .relation 82123
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82123) (rhsResult := 82091)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56118⟩⟩) ⟨55195⟩ 82091) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82124

namespace LeftMerge82132
def owner : Owner := ⟨.program ⟨257⟩, ⟨54257⟩⟩
def mergeEvent : Nat := 82132
def frameStart : Nat := 82043
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events320.exact82105RawTerms
def rightRaw : List Term := Proof.Events320.exact82128RawTerms
def group : MergeGroup := .operator 82105 82128
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 82105) (leftOrdinal := 0)
    (rightResult := 82128) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54255⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82132

namespace LeftMerge82149
def owner : Owner := ⟨.program ⟨257⟩, ⟨54859⟩⟩
def mergeEvent : Nat := 82149
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }
def rhsRaw : List Term := Proof.Events320.exact82146RawTerms
def group : MergeGroup := .relation 82148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82148) (rhsResult := 82146)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82147 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩) (none) 82146) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82149

namespace LeftMerge82150
def owner : Owner := ⟨.program ⟨257⟩, ⟨54859⟩⟩
def mergeEvent : Nat := 82150
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩] } }
def rhsRaw : List Term := Proof.Events320.exact82146RawTerms
def group : MergeGroup := .relation 82148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82148) (rhsResult := 82146)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82147 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩) (none) 82146) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56118⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge82150

namespace LeftMerge82151
def owner : Owner := ⟨.program ⟨257⟩, ⟨54859⟩⟩
def mergeEvent : Nat := 82151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55195⟩⟩] } }
def rhsRaw : List Term := Proof.Events320.exact82146RawTerms
def group : MergeGroup := .relation 82148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 82148) (rhsResult := 82146)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 82147 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54856⟩⟩]⟩) (none) 82146) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨53916⟩⟩], [⟨.program ⟨257⟩, ⟨55195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge82151

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
