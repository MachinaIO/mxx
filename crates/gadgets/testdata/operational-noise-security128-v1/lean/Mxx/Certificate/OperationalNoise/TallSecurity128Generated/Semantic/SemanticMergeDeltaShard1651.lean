import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge267029
def owner : Owner := ⟨.program ⟨257⟩, ⟨42280⟩⟩
def mergeEvent : Nat := 267029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267023RawTerms
def rightRaw : List Term := Proof.Events050.exact12859RawTerms
def group : MergeGroup := .operator 267023 12859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267023) (leftOrdinal := 1)
    (rightResult := 12859) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267029

namespace LeftMerge267030
def owner : Owner := ⟨.program ⟨257⟩, ⟨42280⟩⟩
def mergeEvent : Nat := 267030
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267023RawTerms
def rightRaw : List Term := Proof.Events050.exact12859RawTerms
def group : MergeGroup := .operator 267023 12859
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267023) (leftOrdinal := 0)
    (rightResult := 12859) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267030

namespace LeftMerge267035
def owner : Owner := ⟨.program ⟨257⟩, ⟨14357⟩⟩
def mergeEvent : Nat := 267035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events050.exact12859RawTerms
def rightRaw : List Term := Proof.Events1039.exact266028RawTerms
def group : MergeGroup := .operator 12859 266028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12859) (leftOrdinal := 0)
    (rightResult := 266028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267035

namespace LeftMerge267040
def owner : Owner := ⟨.program ⟨257⟩, ⟨7656⟩⟩
def mergeEvent : Nat := 267040
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }
def leftRaw : List Term := Proof.Events1038.exact265898RawTerms
def rightRaw : List Term := Proof.Events070.exact18123RawTerms
def group : MergeGroup := .operator 265898 18123
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 265898) (leftOrdinal := 0)
    (rightResult := 18123) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267040

namespace LeftMerge267057
def owner : Owner := ⟨.program ⟨257⟩, ⟨14360⟩⟩
def mergeEvent : Nat := 267057
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267051RawTerms
def rightRaw : List Term := Proof.Events070.exact18112RawTerms
def group : MergeGroup := .operator 267051 18112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267051) (leftOrdinal := 1)
    (rightResult := 18112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267057

namespace LeftMerge267059
def owner : Owner := ⟨.program ⟨257⟩, ⟨14360⟩⟩
def mergeEvent : Nat := 267059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def rhsRaw : List Term := Proof.Events070.exact18082RawTerms
def group : MergeGroup := .relation 267058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267058) (rhsResult := 18082)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9559⟩⟩) ⟨7283⟩ 18082) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267059

namespace LeftMerge267060
def owner : Owner := ⟨.program ⟨257⟩, ⟨14360⟩⟩
def mergeEvent : Nat := 267060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267051RawTerms
def rightRaw : List Term := Proof.Events070.exact18112RawTerms
def group : MergeGroup := .operator 267051 18112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267051) (leftOrdinal := 0)
    (rightResult := 18112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267060

namespace LeftMerge267065
def owner : Owner := ⟨.program ⟨257⟩, ⟨42281⟩⟩
def mergeEvent : Nat := 267065
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267061RawTerms
def rightRaw : List Term := Proof.Events1043.exact267031RawTerms
def group : MergeGroup := .operator 267061 267031
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267061) (leftOrdinal := 1)
    (rightResult := 267031) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267065

namespace LeftMerge267073
def owner : Owner := ⟨.program ⟨257⟩, ⟨44209⟩⟩
def mergeEvent : Nat := 267073
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267067RawTerms
def rightRaw : List Term := Proof.Events1042.exact267003RawTerms
def group : MergeGroup := .operator 267067 267003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267067) (leftOrdinal := 1)
    (rightResult := 267003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44208⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267073

namespace LeftMerge267075
def owner : Owner := ⟨.program ⟨257⟩, ⟨44209⟩⟩
def mergeEvent : Nat := 267075
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43739⟩⟩] } }
def rhsRaw : List Term := Proof.Events1042.exact267000RawTerms
def group : MergeGroup := .relation 267074
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267074) (rhsResult := 267000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44208⟩⟩) ⟨43739⟩ 267000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43739⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨43739⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267075

namespace LeftMerge267076
def owner : Owner := ⟨.program ⟨257⟩, ⟨44209⟩⟩
def mergeEvent : Nat := 267076
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267067RawTerms
def rightRaw : List Term := Proof.Events1042.exact267003RawTerms
def group : MergeGroup := .operator 267067 267003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267067) (leftOrdinal := 0)
    (rightResult := 267003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44208⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267076

namespace LeftMerge267090
def owner : Owner := ⟨.program ⟨257⟩, ⟨43149⟩⟩
def mergeEvent : Nat := 267090
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1043.exact267084RawTerms
def group : MergeGroup := .operator 266120 267084
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 267084) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43146⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43146⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267090

namespace LeftMerge267169
def owner : Owner := ⟨.program ⟨257⟩, ⟨42275⟩⟩
def mergeEvent : Nat := 267169
def frameStart : Nat := 267139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1043.exact267165RawTerms
def rightRaw : List Term := Proof.Events1043.exact267162RawTerms
def group : MergeGroup := .operator 267165 267162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267165) (leftOrdinal := 0)
    (rightResult := 267162) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267169

namespace LeftMerge267199
def owner : Owner := ⟨.program ⟨257⟩, ⟨44036⟩⟩
def mergeEvent : Nat := 267199
def frameStart : Nat := 267139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267195RawTerms
def rightRaw : List Term := Proof.Events1043.exact267193RawTerms
def group : MergeGroup := .operator 267195 267193
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267195) (leftOrdinal := 0)
    (rightResult := 267193) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14356⟩⟩, ⟨.program ⟨257⟩, ⟨42274⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267199

namespace LeftMerge267222
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def mergeEvent : Nat := 267222
def frameStart : Nat := 267139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267218RawTerms
def rightRaw : List Term := Proof.Events1043.exact267215RawTerms
def group : MergeGroup := .operator 267218 267215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267218) (leftOrdinal := 0)
    (rightResult := 267215) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267222

namespace LeftMerge267231
def owner : Owner := ⟨.program ⟨257⟩, ⟨44211⟩⟩
def mergeEvent : Nat := 267231
def frameStart : Nat := 267139
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩] } }
def leftRaw : List Term := Proof.Events1043.exact267227RawTerms
def rightRaw : List Term := Proof.Events1043.exact267184RawTerms
def group : MergeGroup := .operator 267227 267184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267227) (leftOrdinal := 0)
    (rightResult := 267184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44208⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44208⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267231

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
