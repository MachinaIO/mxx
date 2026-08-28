import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge276960
def owner : Owner := ⟨.program ⟨257⟩, ⟨44460⟩⟩
def mergeEvent : Nat := 276960
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15575RawTerms
def group : MergeGroup := .relation 276959
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276959) (rhsResult := 15575)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7153⟩⟩) ⟨7042⟩ 15575) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276960

namespace LeftMerge276974
def owner : Owner := ⟨.program ⟨257⟩, ⟨41778⟩⟩
def mergeEvent : Nat := 276974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267752RawTerms
def rightRaw : List Term := Proof.Events1081.exact276968RawTerms
def group : MergeGroup := .operator 267752 276968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267752) (leftOrdinal := 0)
    (rightResult := 276968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276974

namespace LeftMerge276975
def owner : Owner := ⟨.program ⟨257⟩, ⟨41778⟩⟩
def mergeEvent : Nat := 276975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267752RawTerms
def rightRaw : List Term := Proof.Events1081.exact276968RawTerms
def group : MergeGroup := .operator 267752 276968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267752) (leftOrdinal := 1)
    (rightResult := 276968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276975

namespace LeftMerge276977
def owner : Owner := ⟨.program ⟨257⟩, ⟨41778⟩⟩
def mergeEvent : Nat := 276977
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41185⟩⟩] } }
def rhsRaw : List Term := Proof.Events1081.exact276965RawTerms
def group : MergeGroup := .relation 276976
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 276976) (rhsResult := 276965)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41776⟩⟩) ⟨41185⟩ 276965) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨41185⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge276977

namespace LeftMerge276991
def owner : Owner := ⟨.program ⟨257⟩, ⟨40689⟩⟩
def mergeEvent : Nat := 276991
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1081.exact276985RawTerms
def group : MergeGroup := .operator 266120 276985
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 276985) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40686⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge276991

namespace LeftMerge277112
def owner : Owner := ⟨.program ⟨257⟩, ⟨41436⟩⟩
def mergeEvent : Nat := 277112
def frameStart : Nat := 277046
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1082.exact277108RawTerms
def rightRaw : List Term := Proof.Events1082.exact277106RawTerms
def group : MergeGroup := .operator 277108 277106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 277108) (leftOrdinal := 0)
    (rightResult := 277106) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge277112

namespace LeftMerge277124
def owner : Owner := ⟨.program ⟨257⟩, ⟨41777⟩⟩
def mergeEvent : Nat := 277124
def frameStart : Nat := 277046
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩] } }
def leftRaw : List Term := Proof.Events1082.exact277120RawTerms
def rightRaw : List Term := Proof.Events1082.exact277097RawTerms
def group : MergeGroup := .operator 277120 277097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 277120) (leftOrdinal := 0)
    (rightResult := 277097) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41776⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge277124

namespace LeftMerge277125
def owner : Owner := ⟨.program ⟨257⟩, ⟨41777⟩⟩
def mergeEvent : Nat := 277125
def frameStart : Nat := 277046
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩] } }
def leftRaw : List Term := Proof.Events1082.exact277120RawTerms
def rightRaw : List Term := Proof.Events1082.exact277097RawTerms
def group : MergeGroup := .operator 277120 277097
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 277120) (leftOrdinal := 1)
    (rightResult := 277097) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge277125

namespace LeftMerge277127
def owner : Owner := ⟨.program ⟨257⟩, ⟨41777⟩⟩
def mergeEvent : Nat := 277127
def frameStart : Nat := 277046
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41185⟩⟩] } }
def rhsRaw : List Term := Proof.Events1082.exact277094RawTerms
def group : MergeGroup := .relation 277126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 277126) (rhsResult := 277094)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41776⟩⟩) ⟨41185⟩ 277094) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨41185⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge277127

namespace LeftMerge277135
def owner : Owner := ⟨.program ⟨257⟩, ⟨40217⟩⟩
def mergeEvent : Nat := 277135
def frameStart : Nat := 277046
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1082.exact277108RawTerms
def rightRaw : List Term := Proof.Events1082.exact277131RawTerms
def group : MergeGroup := .operator 277108 277131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 277108) (leftOrdinal := 0)
    (rightResult := 277131) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40215⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge277135

namespace LeftMerge277152
def owner : Owner := ⟨.program ⟨257⟩, ⟨40689⟩⟩
def mergeEvent : Nat := 277152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }
def rhsRaw : List Term := Proof.Events1082.exact277149RawTerms
def group : MergeGroup := .relation 277151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 277151) (rhsResult := 277149)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 277150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩]⟩) (none) 277149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge277152

namespace LeftMerge277153
def owner : Owner := ⟨.program ⟨257⟩, ⟨40689⟩⟩
def mergeEvent : Nat := 277153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩] } }
def rhsRaw : List Term := Proof.Events1082.exact277149RawTerms
def group : MergeGroup := .relation 277151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 277151) (rhsResult := 277149)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 277150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩]⟩) (none) 277149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge277153

namespace LeftMerge277154
def owner : Owner := ⟨.program ⟨257⟩, ⟨40689⟩⟩
def mergeEvent : Nat := 277154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41185⟩⟩] } }
def rhsRaw : List Term := Proof.Events1082.exact277149RawTerms
def group : MergeGroup := .relation 277151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 277151) (rhsResult := 277149)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 277150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩]⟩) (none) 277149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨41185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge277154

namespace LeftMerge277155
def owner : Owner := ⟨.program ⟨257⟩, ⟨40689⟩⟩
def mergeEvent : Nat := 277155
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1082.exact277149RawTerms
def group : MergeGroup := .relation 277151
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 277151) (rhsResult := 277149)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 277150 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40686⟩⟩]⟩) (none) 277149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge277155

namespace LeftMerge277160
def owner : Owner := ⟨.program ⟨257⟩, ⟨41779⟩⟩
def mergeEvent : Nat := 277160
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩] } }
def leftRaw : List Term := Proof.Events1082.exact277156RawTerms
def rightRaw : List Term := Proof.Events1081.exact276978RawTerms
def group : MergeGroup := .operator 277156 276978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 277156) (leftOrdinal := 0)
    (rightResult := 276978) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge277160

namespace LeftMerge277161
def owner : Owner := ⟨.program ⟨257⟩, ⟨41779⟩⟩
def mergeEvent : Nat := 277161
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41185⟩⟩] } }
def leftRaw : List Term := Proof.Events1082.exact277156RawTerms
def rightRaw : List Term := Proof.Events1081.exact276978RawTerms
def group : MergeGroup := .operator 277156 276978
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 277156) (leftOrdinal := 2)
    (rightResult := 276978) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41185⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41185⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨41185⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge277161

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
