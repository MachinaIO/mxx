import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge121936
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 121936
def frameStart : Nat := 121853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events476.exact121932RawTerms
def rightRaw : List Term := Proof.Events476.exact121929RawTerms
def group : MergeGroup := .operator 121932 121929
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121932) (leftOrdinal := 0)
    (rightResult := 121929) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121936

namespace LeftMerge121945
def owner : Owner := ⟨.program ⟨257⟩, ⟨38898⟩⟩
def mergeEvent : Nat := 121945
def frameStart : Nat := 121853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩] } }
def leftRaw : List Term := Proof.Events476.exact121941RawTerms
def rightRaw : List Term := Proof.Events476.exact121898RawTerms
def group : MergeGroup := .operator 121941 121898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121941) (leftOrdinal := 0)
    (rightResult := 121898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38895⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121945

namespace LeftMerge121946
def owner : Owner := ⟨.program ⟨257⟩, ⟨38898⟩⟩
def mergeEvent : Nat := 121946
def frameStart : Nat := 121853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩] } }
def leftRaw : List Term := Proof.Events476.exact121941RawTerms
def rightRaw : List Term := Proof.Events476.exact121898RawTerms
def group : MergeGroup := .operator 121941 121898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121941) (leftOrdinal := 1)
    (rightResult := 121898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38895⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121946

namespace LeftMerge121948
def owner : Owner := ⟨.program ⟨257⟩, ⟨38898⟩⟩
def mergeEvent : Nat := 121948
def frameStart : Nat := 121853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38405⟩⟩] } }
def rhsRaw : List Term := Proof.Events476.exact121895RawTerms
def group : MergeGroup := .relation 121947
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121947) (rhsResult := 121895)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38895⟩⟩) ⟨38405⟩ 121895) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38405⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨38405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121948

namespace LeftMerge121956
def owner : Owner := ⟨.program ⟨257⟩, ⟨37398⟩⟩
def mergeEvent : Nat := 121956
def frameStart : Nat := 121853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37396⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events476.exact121909RawTerms
def rightRaw : List Term := Proof.Events476.exact121952RawTerms
def group : MergeGroup := .operator 121909 121952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121909) (leftOrdinal := 0)
    (rightResult := 121952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121956

namespace LeftMerge121973
def owner : Owner := ⟨.program ⟨257⟩, ⟨37832⟩⟩
def mergeEvent : Nat := 121973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events476.exact121970RawTerms
def group : MergeGroup := .relation 121972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121972) (rhsResult := 121970)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121971 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩) (none) 121970) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121973

namespace LeftMerge121974
def owner : Owner := ⟨.program ⟨257⟩, ⟨37832⟩⟩
def mergeEvent : Nat := 121974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩] } }
def rhsRaw : List Term := Proof.Events476.exact121970RawTerms
def group : MergeGroup := .relation 121972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121972) (rhsResult := 121970)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121971 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩) (none) 121970) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121974

namespace LeftMerge121975
def owner : Owner := ⟨.program ⟨257⟩, ⟨37832⟩⟩
def mergeEvent : Nat := 121975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38405⟩⟩] } }
def rhsRaw : List Term := Proof.Events476.exact121970RawTerms
def group : MergeGroup := .relation 121972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121972) (rhsResult := 121970)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121971 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩) (none) 121970) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38405⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨38405⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121975

namespace LeftMerge121976
def owner : Owner := ⟨.program ⟨257⟩, ⟨37832⟩⟩
def mergeEvent : Nat := 121976
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events476.exact121970RawTerms
def group : MergeGroup := .relation 121972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121972) (rhsResult := 121970)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 121971 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37829⟩⟩]⟩) (none) 121970) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37396⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121976

namespace LeftMerge121981
def owner : Owner := ⟨.program ⟨257⟩, ⟨38897⟩⟩
def mergeEvent : Nat := 121981
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38405⟩⟩] } }
def leftRaw : List Term := Proof.Events476.exact121977RawTerms
def rightRaw : List Term := Proof.Events475.exact121791RawTerms
def group : MergeGroup := .operator 121977 121791
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121977) (leftOrdinal := 2)
    (rightResult := 121791) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38405⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨13821⟩⟩, ⟨.program ⟨257⟩, ⟨37018⟩⟩], [⟨.program ⟨257⟩, ⟨38405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121981

namespace LeftMerge121982
def owner : Owner := ⟨.program ⟨257⟩, ⟨38897⟩⟩
def mergeEvent : Nat := 121982
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩] } }
def leftRaw : List Term := Proof.Events476.exact121977RawTerms
def rightRaw : List Term := Proof.Events475.exact121791RawTerms
def group : MergeGroup := .operator 121977 121791
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121977) (leftOrdinal := 1)
    (rightResult := 121791) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38895⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121982

namespace LeftMerge121990
def owner : Owner := ⟨.program ⟨257⟩, ⟨39211⟩⟩
def mergeEvent : Nat := 121990
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39209⟩⟩] } }
def leftRaw : List Term := Proof.Events476.exact121984RawTerms
def rightRaw : List Term := Proof.Events475.exact121707RawTerms
def group : MergeGroup := .operator 121984 121707
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121984) (leftOrdinal := 0)
    (rightResult := 121707) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39209⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39209⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge121990

namespace LeftMerge121991
def owner : Owner := ⟨.program ⟨257⟩, ⟨39211⟩⟩
def mergeEvent : Nat := 121991
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39209⟩⟩] } }
def leftRaw : List Term := Proof.Events476.exact121984RawTerms
def rightRaw : List Term := Proof.Events475.exact121707RawTerms
def group : MergeGroup := .operator 121984 121707
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 121984) (leftOrdinal := 1)
    (rightResult := 121707) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39209⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39209⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121991

namespace LeftMerge121993
def owner : Owner := ⟨.program ⟨257⟩, ⟨39211⟩⟩
def mergeEvent : Nat := 121993
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38545⟩⟩] } }
def rhsRaw : List Term := Proof.Events475.exact121704RawTerms
def group : MergeGroup := .relation 121992
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 121992) (rhsResult := 121704)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39209⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39209⟩⟩) ⟨38545⟩ 121704) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38545⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨38545⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge121993

namespace LeftMerge122007
def owner : Owner := ⟨.program ⟨257⟩, ⟨38099⟩⟩
def mergeEvent : Nat := 122007
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38096⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events476.exact122001RawTerms
def group : MergeGroup := .operator 119870 122001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 122001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38096⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38096⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122007

namespace LeftMerge122128
def owner : Owner := ⟨.program ⟨257⟩, ⟨38772⟩⟩
def mergeEvent : Nat := 122128
def frameStart : Nat := 122062
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37396⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events477.exact122124RawTerms
def rightRaw : List Term := Proof.Events477.exact122122RawTerms
def group : MergeGroup := .operator 122124 122122
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 122124) (leftOrdinal := 0)
    (rightResult := 122122) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37396⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37396⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge122128

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
