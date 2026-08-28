import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge86122
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86122
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 7) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86122

namespace LeftMerge86123
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86123
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 6) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86123

namespace LeftMerge86124
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 5) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86124

namespace LeftMerge86125
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86125
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 4) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86125

namespace LeftMerge86126
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86126
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86126

namespace LeftMerge86127
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86127
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86127

namespace LeftMerge86128
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86128
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86128

namespace LeftMerge86129
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86129
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge86129

namespace LeftMerge86130
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86130
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 30) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48441⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48441⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86130

namespace LeftMerge86131
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86131
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 29) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45761⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45761⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86131

namespace LeftMerge86132
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 28) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86132

namespace LeftMerge86133
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 27) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86133

namespace LeftMerge86134
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 26) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86134

namespace LeftMerge86135
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 25) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35041⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35041⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86135

namespace LeftMerge86136
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 23) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29377⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86136

namespace LeftMerge86137
def owner : Owner := ⟨.program ⟨257⟩, ⟨68433⟩⟩
def mergeEvent : Nat := 86137
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events336.exact86108RawTerms
def group : MergeGroup := .relation 86110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 86110) (rhsResult := 86108)
    (sourceTermOrdinal := 22) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 86109 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68430⟩⟩]⟩) (none) 86108) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26697⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨26697⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86137

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
