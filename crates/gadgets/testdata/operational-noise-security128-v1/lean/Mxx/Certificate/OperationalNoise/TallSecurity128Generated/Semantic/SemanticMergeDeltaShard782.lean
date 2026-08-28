import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge129946
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129946
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54065⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129945
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129945) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨54065⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129946

namespace LeftMerge129947
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129947
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51085⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 30)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51085⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129947

namespace LeftMerge129949
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129949
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51085⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129948
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129948) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨51085⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129949

namespace LeftMerge129950
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129950
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 23)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129950

namespace LeftMerge129952
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129952
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32030⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129951) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨32030⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129952

namespace LeftMerge129953
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129953
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 20)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129953

namespace LeftMerge129955
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129955
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129954
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129954) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨22010⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129955

namespace LeftMerge129956
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129956
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18790⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 19)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18790⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129956

namespace LeftMerge129958
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129958
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18790⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129957) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129958

namespace LeftMerge129959
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129959
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 18)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129959

namespace LeftMerge129961
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129961
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129960
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129960) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15971⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129961

namespace LeftMerge129969
def owner : Owner := ⟨.program ⟨257⟩, ⟨67384⟩⟩
def mergeEvent : Nat := 129969
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events506.exact129738RawTerms
def rightRaw : List Term := Proof.Events507.exact129965RawTerms
def group : MergeGroup := .operator 129738 129965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129738) (leftOrdinal := 0)
    (rightResult := 129965) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67382⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨67382⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129969

namespace LeftMerge129986
def owner : Owner := ⟨.program ⟨257⟩, ⟨68333⟩⟩
def mergeEvent : Nat := 129986
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }
def rhsRaw : List Term := Proof.Events507.exact129983RawTerms
def group : MergeGroup := .relation 129985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129985) (rhsResult := 129983)
    (sourceTermOrdinal := 18) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 129984 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩]⟩) (none) 129983) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129986

namespace LeftMerge129987
def owner : Owner := ⟨.program ⟨257⟩, ⟨68333⟩⟩
def mergeEvent : Nat := 129987
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def rhsRaw : List Term := Proof.Events507.exact129983RawTerms
def group : MergeGroup := .relation 129985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129985) (rhsResult := 129983)
    (sourceTermOrdinal := 17) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 129984 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩]⟩) (none) 129983) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129987

namespace LeftMerge129988
def owner : Owner := ⟨.program ⟨257⟩, ⟨68333⟩⟩
def mergeEvent : Nat := 129988
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def rhsRaw : List Term := Proof.Events507.exact129983RawTerms
def group : MergeGroup := .relation 129985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129985) (rhsResult := 129983)
    (sourceTermOrdinal := 16) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 129984 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩]⟩) (none) 129983) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129988

namespace LeftMerge129989
def owner : Owner := ⟨.program ⟨257⟩, ⟨68333⟩⟩
def mergeEvent : Nat := 129989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def rhsRaw : List Term := Proof.Events507.exact129983RawTerms
def group : MergeGroup := .relation 129985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129985) (rhsResult := 129983)
    (sourceTermOrdinal := 15) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 129984 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68330⟩⟩]⟩) (none) 129983) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129989

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
