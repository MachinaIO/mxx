import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge78100
def owner : Owner := ⟨.program ⟨257⟩, ⟨37932⟩⟩
def mergeEvent : Nat := 78100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38465⟩⟩] } }
def rhsRaw : List Term := Proof.Events305.exact78095RawTerms
def group : MergeGroup := .relation 78097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78097) (rhsResult := 78095)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 78096 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩) (none) 78095) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38465⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78100

namespace LeftMerge78101
def owner : Owner := ⟨.program ⟨257⟩, ⟨37932⟩⟩
def mergeEvent : Nat := 78101
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events305.exact78095RawTerms
def group : MergeGroup := .relation 78097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78097) (rhsResult := 78095)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 78096 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩) (none) 78095) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78101

namespace LeftMerge78106
def owner : Owner := ⟨.program ⟨257⟩, ⟨39007⟩⟩
def mergeEvent : Nat := 78106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38465⟩⟩] } }
def leftRaw : List Term := Proof.Events305.exact78102RawTerms
def rightRaw : List Term := Proof.Events304.exact77916RawTerms
def group : MergeGroup := .operator 78102 77916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78102) (leftOrdinal := 2)
    (rightResult := 77916) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38465⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38465⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78106

namespace LeftMerge78107
def owner : Owner := ⟨.program ⟨257⟩, ⟨39007⟩⟩
def mergeEvent : Nat := 78107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩] } }
def leftRaw : List Term := Proof.Events305.exact78102RawTerms
def rightRaw : List Term := Proof.Events304.exact77916RawTerms
def group : MergeGroup := .operator 78102 77916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78102) (leftOrdinal := 1)
    (rightResult := 77916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78107

namespace LeftMerge78115
def owner : Owner := ⟨.program ⟨257⟩, ⟨39461⟩⟩
def mergeEvent : Nat := 78115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩] } }
def leftRaw : List Term := Proof.Events305.exact78109RawTerms
def rightRaw : List Term := Proof.Events304.exact77832RawTerms
def group : MergeGroup := .operator 78109 77832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78109) (leftOrdinal := 0)
    (rightResult := 77832) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78115

namespace LeftMerge78116
def owner : Owner := ⟨.program ⟨257⟩, ⟨39461⟩⟩
def mergeEvent : Nat := 78116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩] } }
def leftRaw : List Term := Proof.Events305.exact78109RawTerms
def rightRaw : List Term := Proof.Events304.exact77832RawTerms
def group : MergeGroup := .operator 78109 77832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78109) (leftOrdinal := 1)
    (rightResult := 77832) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78116

namespace LeftMerge78118
def owner : Owner := ⟨.program ⟨257⟩, ⟨39461⟩⟩
def mergeEvent : Nat := 78118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38635⟩⟩] } }
def rhsRaw : List Term := Proof.Events304.exact77829RawTerms
def group : MergeGroup := .relation 78117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78117) (rhsResult := 77829)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39459⟩⟩) ⟨38635⟩ 77829) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38635⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78118

namespace LeftMerge78132
def owner : Owner := ⟨.program ⟨257⟩, ⟨38299⟩⟩
def mergeEvent : Nat := 78132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events305.exact78126RawTerms
def group : MergeGroup := .operator 75995 78126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 78126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38296⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78132

namespace LeftMerge78253
def owner : Owner := ⟨.program ⟨257⟩, ⟨38812⟩⟩
def mergeEvent : Nat := 78253
def frameStart : Nat := 78187
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events305.exact78249RawTerms
def rightRaw : List Term := Proof.Events305.exact78247RawTerms
def group : MergeGroup := .operator 78249 78247
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78249) (leftOrdinal := 0)
    (rightResult := 78247) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78253

namespace LeftMerge78265
def owner : Owner := ⟨.program ⟨257⟩, ⟨39460⟩⟩
def mergeEvent : Nat := 78265
def frameStart : Nat := 78187
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩] } }
def leftRaw : List Term := Proof.Events305.exact78261RawTerms
def rightRaw : List Term := Proof.Events305.exact78238RawTerms
def group : MergeGroup := .operator 78261 78238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78261) (leftOrdinal := 0)
    (rightResult := 78238) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39459⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78265

namespace LeftMerge78266
def owner : Owner := ⟨.program ⟨257⟩, ⟨39460⟩⟩
def mergeEvent : Nat := 78266
def frameStart : Nat := 78187
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩] } }
def leftRaw : List Term := Proof.Events305.exact78261RawTerms
def rightRaw : List Term := Proof.Events305.exact78238RawTerms
def group : MergeGroup := .operator 78261 78238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78261) (leftOrdinal := 1)
    (rightResult := 78238) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78266

namespace LeftMerge78268
def owner : Owner := ⟨.program ⟨257⟩, ⟨39460⟩⟩
def mergeEvent : Nat := 78268
def frameStart : Nat := 78187
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38635⟩⟩] } }
def rhsRaw : List Term := Proof.Events305.exact78235RawTerms
def group : MergeGroup := .relation 78267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78267) (rhsResult := 78235)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39459⟩⟩) ⟨38635⟩ 78235) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38635⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78268

namespace LeftMerge78276
def owner : Owner := ⟨.program ⟨257⟩, ⟨37722⟩⟩
def mergeEvent : Nat := 78276
def frameStart : Nat := 78187
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events305.exact78249RawTerms
def rightRaw : List Term := Proof.Events305.exact78272RawTerms
def group : MergeGroup := .operator 78249 78272
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78249) (leftOrdinal := 0)
    (rightResult := 78272) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37721⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37721⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78276

namespace LeftMerge78293
def owner : Owner := ⟨.program ⟨257⟩, ⟨38299⟩⟩
def mergeEvent : Nat := 78293
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }
def rhsRaw : List Term := Proof.Events305.exact78290RawTerms
def group : MergeGroup := .relation 78292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78292) (rhsResult := 78290)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 78291 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩) (none) 78290) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78293

namespace LeftMerge78294
def owner : Owner := ⟨.program ⟨257⟩, ⟨38299⟩⟩
def mergeEvent : Nat := 78294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩] } }
def rhsRaw : List Term := Proof.Events305.exact78290RawTerms
def group : MergeGroup := .relation 78292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78292) (rhsResult := 78290)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 78291 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩) (none) 78290) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39459⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78294

namespace LeftMerge78295
def owner : Owner := ⟨.program ⟨257⟩, ⟨38299⟩⟩
def mergeEvent : Nat := 78295
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38635⟩⟩] } }
def rhsRaw : List Term := Proof.Events305.exact78290RawTerms
def group : MergeGroup := .relation 78292
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78292) (rhsResult := 78290)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 78291 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38296⟩⟩]⟩) (none) 78290) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38635⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38635⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78295

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
