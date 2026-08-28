import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge77898
def owner : Owner := ⟨.program ⟨257⟩, ⟨13975⟩⟩
def mergeEvent : Nat := 77898
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def rhsRaw : List Term := Proof.Events074.exact19084RawTerms
def group : MergeGroup := .relation 77897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77897) (rhsResult := 19084)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9553⟩⟩) ⟨7281⟩ 19084) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77898

namespace LeftMerge77899
def owner : Owner := ⟨.program ⟨257⟩, ⟨13975⟩⟩
def mergeEvent : Nat := 77899
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact77890RawTerms
def rightRaw : List Term := Proof.Events074.exact19114RawTerms
def group : MergeGroup := .operator 77890 19114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77890) (leftOrdinal := 0)
    (rightResult := 19114) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77899

namespace LeftMerge77904
def owner : Owner := ⟨.program ⟨257⟩, ⟨37265⟩⟩
def mergeEvent : Nat := 77904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact77900RawTerms
def rightRaw : List Term := Proof.Events304.exact77870RawTerms
def group : MergeGroup := .operator 77900 77870
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77900) (leftOrdinal := 1)
    (rightResult := 77870) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77904

namespace LeftMerge77912
def owner : Owner := ⟨.program ⟨257⟩, ⟨39006⟩⟩
def mergeEvent : Nat := 77912
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact77906RawTerms
def rightRaw : List Term := Proof.Events304.exact77842RawTerms
def group : MergeGroup := .operator 77906 77842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77906) (leftOrdinal := 1)
    (rightResult := 77842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39005⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77912

namespace LeftMerge77914
def owner : Owner := ⟨.program ⟨257⟩, ⟨39006⟩⟩
def mergeEvent : Nat := 77914
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38465⟩⟩] } }
def rhsRaw : List Term := Proof.Events304.exact77839RawTerms
def group : MergeGroup := .relation 77913
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77913) (rhsResult := 77839)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39005⟩⟩) ⟨38465⟩ 77839) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38465⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77914

namespace LeftMerge77915
def owner : Owner := ⟨.program ⟨257⟩, ⟨39006⟩⟩
def mergeEvent : Nat := 77915
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact77906RawTerms
def rightRaw : List Term := Proof.Events304.exact77842RawTerms
def group : MergeGroup := .operator 77906 77842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77906) (leftOrdinal := 0)
    (rightResult := 77842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39005⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77915

namespace LeftMerge77929
def owner : Owner := ⟨.program ⟨257⟩, ⟨37932⟩⟩
def mergeEvent : Nat := 77929
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events304.exact77923RawTerms
def group : MergeGroup := .operator 75995 77923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 77923) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37929⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77929

namespace LeftMerge78008
def owner : Owner := ⟨.program ⟨257⟩, ⟨37259⟩⟩
def mergeEvent : Nat := 78008
def frameStart : Nat := 77978
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events304.exact78004RawTerms
def rightRaw : List Term := Proof.Events304.exact78001RawTerms
def group : MergeGroup := .operator 78004 78001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78004) (leftOrdinal := 0)
    (rightResult := 78001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78008

namespace LeftMerge78038
def owner : Owner := ⟨.program ⟨257⟩, ⟨38732⟩⟩
def mergeEvent : Nat := 78038
def frameStart : Nat := 77978
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact78034RawTerms
def rightRaw : List Term := Proof.Events304.exact78032RawTerms
def group : MergeGroup := .operator 78034 78032
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78034) (leftOrdinal := 0)
    (rightResult := 78032) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78038

namespace LeftMerge78061
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 78061
def frameStart : Nat := 77978
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact78057RawTerms
def rightRaw : List Term := Proof.Events304.exact78054RawTerms
def group : MergeGroup := .operator 78057 78054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78057) (leftOrdinal := 0)
    (rightResult := 78054) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78061

namespace LeftMerge78070
def owner : Owner := ⟨.program ⟨257⟩, ⟨39008⟩⟩
def mergeEvent : Nat := 78070
def frameStart : Nat := 77978
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact78066RawTerms
def rightRaw : List Term := Proof.Events304.exact78023RawTerms
def group : MergeGroup := .operator 78066 78023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78066) (leftOrdinal := 0)
    (rightResult := 78023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39005⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78070

namespace LeftMerge78071
def owner : Owner := ⟨.program ⟨257⟩, ⟨39008⟩⟩
def mergeEvent : Nat := 78071
def frameStart : Nat := 77978
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact78066RawTerms
def rightRaw : List Term := Proof.Events304.exact78023RawTerms
def group : MergeGroup := .operator 78066 78023
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78066) (leftOrdinal := 1)
    (rightResult := 78023) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39005⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78071

namespace LeftMerge78073
def owner : Owner := ⟨.program ⟨257⟩, ⟨39008⟩⟩
def mergeEvent : Nat := 78073
def frameStart : Nat := 77978
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38465⟩⟩] } }
def rhsRaw : List Term := Proof.Events304.exact78020RawTerms
def group : MergeGroup := .relation 78072
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78072) (rhsResult := 78020)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39005⟩⟩) ⟨38465⟩ 78020) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38465⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨38465⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78073

namespace LeftMerge78081
def owner : Owner := ⟨.program ⟨257⟩, ⟨37478⟩⟩
def mergeEvent : Nat := 78081
def frameStart : Nat := 77978
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact78034RawTerms
def rightRaw : List Term := Proof.Events304.exact78077RawTerms
def group : MergeGroup := .operator 78034 78077
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78034) (leftOrdinal := 0)
    (rightResult := 78077) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78081

namespace LeftMerge78098
def owner : Owner := ⟨.program ⟨257⟩, ⟨37932⟩⟩
def mergeEvent : Nat := 78098
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events305.exact78095RawTerms
def group : MergeGroup := .relation 78097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78097) (rhsResult := 78095)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 78096 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩) (none) 78095) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge78098

namespace LeftMerge78099
def owner : Owner := ⟨.program ⟨257⟩, ⟨37932⟩⟩
def mergeEvent : Nat := 78099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩] } }
def rhsRaw : List Term := Proof.Events305.exact78095RawTerms
def group : MergeGroup := .relation 78097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 78097) (rhsResult := 78095)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 78096 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37929⟩⟩]⟩) (none) 78095) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39005⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge78099

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
