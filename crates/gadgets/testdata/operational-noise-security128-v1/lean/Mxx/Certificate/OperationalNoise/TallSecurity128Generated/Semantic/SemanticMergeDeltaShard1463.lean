import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge237008
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 237008
def frameStart : Nat := 236925
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact237004RawTerms
def rightRaw : List Term := Proof.Events925.exact237001RawTerms
def group : MergeGroup := .operator 237004 237001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237004) (leftOrdinal := 0)
    (rightResult := 237001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237008

namespace LeftMerge237017
def owner : Owner := ⟨.program ⟨257⟩, ⟨49640⟩⟩
def mergeEvent : Nat := 237017
def frameStart : Nat := 236925
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact237013RawTerms
def rightRaw : List Term := Proof.Events925.exact236970RawTerms
def group : MergeGroup := .operator 237013 236970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237013) (leftOrdinal := 0)
    (rightResult := 236970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49637⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237017

namespace LeftMerge237018
def owner : Owner := ⟨.program ⟨257⟩, ⟨49640⟩⟩
def mergeEvent : Nat := 237018
def frameStart : Nat := 236925
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact237013RawTerms
def rightRaw : List Term := Proof.Events925.exact236970RawTerms
def group : MergeGroup := .operator 237013 236970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237013) (leftOrdinal := 1)
    (rightResult := 236970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49637⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237018

namespace LeftMerge237020
def owner : Owner := ⟨.program ⟨257⟩, ⟨49640⟩⟩
def mergeEvent : Nat := 237020
def frameStart : Nat := 236925
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49137⟩⟩] } }
def rhsRaw : List Term := Proof.Events925.exact236967RawTerms
def group : MergeGroup := .relation 237019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237019) (rhsResult := 236967)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49637⟩⟩) ⟨49137⟩ 236967) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49137⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], [⟨.program ⟨257⟩, ⟨49137⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237020

namespace LeftMerge237028
def owner : Owner := ⟨.program ⟨257⟩, ⟨48134⟩⟩
def mergeEvent : Nat := 237028
def frameStart : Nat := 236925
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48132⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236981RawTerms
def rightRaw : List Term := Proof.Events925.exact237024RawTerms
def group : MergeGroup := .operator 236981 237024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236981) (leftOrdinal := 0)
    (rightResult := 237024) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48132⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237028

namespace LeftMerge237045
def owner : Owner := ⟨.program ⟨257⟩, ⟨48572⟩⟩
def mergeEvent : Nat := 237045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events925.exact237042RawTerms
def group : MergeGroup := .relation 237044
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237044) (rhsResult := 237042)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 237043 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48569⟩⟩]⟩) (none) 237042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237045

namespace LeftMerge237046
def owner : Owner := ⟨.program ⟨257⟩, ⟨48572⟩⟩
def mergeEvent : Nat := 237046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩] } }
def rhsRaw : List Term := Proof.Events925.exact237042RawTerms
def group : MergeGroup := .relation 237044
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237044) (rhsResult := 237042)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 237043 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48569⟩⟩]⟩) (none) 237042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237046

namespace LeftMerge237047
def owner : Owner := ⟨.program ⟨257⟩, ⟨48572⟩⟩
def mergeEvent : Nat := 237047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49137⟩⟩] } }
def rhsRaw : List Term := Proof.Events925.exact237042RawTerms
def group : MergeGroup := .relation 237044
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237044) (rhsResult := 237042)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 237043 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48569⟩⟩]⟩) (none) 237042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49137⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], [⟨.program ⟨257⟩, ⟨49137⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237047

namespace LeftMerge237048
def owner : Owner := ⟨.program ⟨257⟩, ⟨48572⟩⟩
def mergeEvent : Nat := 237048
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events925.exact237042RawTerms
def group : MergeGroup := .relation 237044
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237044) (rhsResult := 237042)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48569⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 237043 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48569⟩⟩]⟩) (none) 237042) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48132⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237048

namespace LeftMerge237053
def owner : Owner := ⟨.program ⟨257⟩, ⟨49639⟩⟩
def mergeEvent : Nat := 237053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49137⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact237049RawTerms
def rightRaw : List Term := Proof.Events925.exact236852RawTerms
def group : MergeGroup := .operator 237049 236852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237049) (leftOrdinal := 2)
    (rightResult := 236852) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49137⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49137⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨15051⟩⟩, ⟨.program ⟨257⟩, ⟨47786⟩⟩], [⟨.program ⟨257⟩, ⟨49137⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237053

namespace LeftMerge237054
def owner : Owner := ⟨.program ⟨257⟩, ⟨49639⟩⟩
def mergeEvent : Nat := 237054
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact237049RawTerms
def rightRaw : List Term := Proof.Events925.exact236852RawTerms
def group : MergeGroup := .operator 237049 236852
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237049) (leftOrdinal := 1)
    (rightResult := 236852) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49637⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237054

namespace LeftMerge237062
def owner : Owner := ⟨.program ⟨257⟩, ⟨49981⟩⟩
def mergeEvent : Nat := 237062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩] } }
def leftRaw : List Term := Proof.Events926.exact237056RawTerms
def rightRaw : List Term := Proof.Events924.exact236763RawTerms
def group : MergeGroup := .operator 237056 236763
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237056) (leftOrdinal := 0)
    (rightResult := 236763) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237062

namespace LeftMerge237063
def owner : Owner := ⟨.program ⟨257⟩, ⟨49981⟩⟩
def mergeEvent : Nat := 237063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩] } }
def leftRaw : List Term := Proof.Events926.exact237056RawTerms
def rightRaw : List Term := Proof.Events924.exact236763RawTerms
def group : MergeGroup := .operator 237056 236763
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237056) (leftOrdinal := 1)
    (rightResult := 236763) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237063

namespace LeftMerge237065
def owner : Owner := ⟨.program ⟨257⟩, ⟨49981⟩⟩
def mergeEvent : Nat := 237065
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49283⟩⟩] } }
def rhsRaw : List Term := Proof.Events924.exact236760RawTerms
def group : MergeGroup := .relation 237064
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237064) (rhsResult := 236760)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49979⟩⟩) ⟨49283⟩ 236760) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49283⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨49283⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237065

namespace LeftMerge237079
def owner : Owner := ⟨.program ⟨257⟩, ⟨48859⟩⟩
def mergeEvent : Nat := 237079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48856⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events926.exact237073RawTerms
def group : MergeGroup := .operator 236870 237073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 237073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48856⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48856⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237079

namespace LeftMerge237200
def owner : Owner := ⟨.program ⟨257⟩, ⟨49500⟩⟩
def mergeEvent : Nat := 237200
def frameStart : Nat := 237134
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48132⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events926.exact237196RawTerms
def rightRaw : List Term := Proof.Events926.exact237194RawTerms
def group : MergeGroup := .operator 237196 237194
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237196) (leftOrdinal := 0)
    (rightResult := 237194) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48132⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48132⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237200

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
