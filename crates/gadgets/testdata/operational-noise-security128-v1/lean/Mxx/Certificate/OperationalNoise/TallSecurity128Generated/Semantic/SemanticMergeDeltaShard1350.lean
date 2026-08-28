import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge220108
def owner : Owner := ⟨.program ⟨257⟩, ⟨61886⟩⟩
def mergeEvent : Nat := 220108
def frameStart : Nat := 220030
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩] } }
def leftRaw : List Term := Proof.Events859.exact220104RawTerms
def rightRaw : List Term := Proof.Events859.exact220081RawTerms
def group : MergeGroup := .operator 220104 220081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220104) (leftOrdinal := 0)
    (rightResult := 220081) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61885⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220108

namespace LeftMerge220109
def owner : Owner := ⟨.program ⟨257⟩, ⟨61886⟩⟩
def mergeEvent : Nat := 220109
def frameStart : Nat := 220030
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩] } }
def leftRaw : List Term := Proof.Events859.exact220104RawTerms
def rightRaw : List Term := Proof.Events859.exact220081RawTerms
def group : MergeGroup := .operator 220104 220081
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220104) (leftOrdinal := 1)
    (rightResult := 220081) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61885⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220109

namespace LeftMerge220111
def owner : Owner := ⟨.program ⟨257⟩, ⟨61886⟩⟩
def mergeEvent : Nat := 220111
def frameStart : Nat := 220030
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61100⟩⟩] } }
def rhsRaw : List Term := Proof.Events859.exact220078RawTerms
def group : MergeGroup := .relation 220110
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220110) (rhsResult := 220078)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61885⟩⟩) ⟨61100⟩ 220078) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61100⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨61100⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220111

namespace LeftMerge220119
def owner : Owner := ⟨.program ⟨257⟩, ⟨60108⟩⟩
def mergeEvent : Nat := 220119
def frameStart : Nat := 220030
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events859.exact220092RawTerms
def rightRaw : List Term := Proof.Events859.exact220115RawTerms
def group : MergeGroup := .operator 220092 220115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220092) (leftOrdinal := 0)
    (rightResult := 220115) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220119

namespace LeftMerge220136
def owner : Owner := ⟨.program ⟨257⟩, ⟨60695⟩⟩
def mergeEvent : Nat := 220136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }
def rhsRaw : List Term := Proof.Events859.exact220133RawTerms
def group : MergeGroup := .relation 220135
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220135) (rhsResult := 220133)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 220134 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60692⟩⟩]⟩) (none) 220133) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220136

namespace LeftMerge220137
def owner : Owner := ⟨.program ⟨257⟩, ⟨60695⟩⟩
def mergeEvent : Nat := 220137
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩] } }
def rhsRaw : List Term := Proof.Events859.exact220133RawTerms
def group : MergeGroup := .relation 220135
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220135) (rhsResult := 220133)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 220134 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60692⟩⟩]⟩) (none) 220133) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220137

namespace LeftMerge220138
def owner : Owner := ⟨.program ⟨257⟩, ⟨60695⟩⟩
def mergeEvent : Nat := 220138
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61100⟩⟩] } }
def rhsRaw : List Term := Proof.Events859.exact220133RawTerms
def group : MergeGroup := .relation 220135
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220135) (rhsResult := 220133)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 220134 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60692⟩⟩]⟩) (none) 220133) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61100⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨61100⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220138

namespace LeftMerge220139
def owner : Owner := ⟨.program ⟨257⟩, ⟨60695⟩⟩
def mergeEvent : Nat := 220139
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events859.exact220133RawTerms
def group : MergeGroup := .relation 220135
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220135) (rhsResult := 220133)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 220134 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60692⟩⟩]⟩) (none) 220133) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220139

namespace LeftMerge220144
def owner : Owner := ⟨.program ⟨257⟩, ⟨61888⟩⟩
def mergeEvent : Nat := 220144
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩] } }
def leftRaw : List Term := Proof.Events859.exact220140RawTerms
def rightRaw : List Term := Proof.Events859.exact219962RawTerms
def group : MergeGroup := .operator 220140 219962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220140) (leftOrdinal := 0)
    (rightResult := 219962) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61885⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220144

namespace LeftMerge220145
def owner : Owner := ⟨.program ⟨257⟩, ⟨61888⟩⟩
def mergeEvent : Nat := 220145
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61100⟩⟩] } }
def leftRaw : List Term := Proof.Events859.exact220140RawTerms
def rightRaw : List Term := Proof.Events859.exact219962RawTerms
def group : MergeGroup := .operator 220140 219962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220140) (leftOrdinal := 2)
    (rightResult := 219962) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61100⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61100⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨59828⟩⟩], [⟨.program ⟨257⟩, ⟨61100⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220145

namespace LeftMerge220153
def owner : Owner := ⟨.program ⟨257⟩, ⟨61889⟩⟩
def mergeEvent : Nat := 220153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }
def leftRaw : List Term := Proof.Events859.exact220147RawTerms
def rightRaw : List Term := Proof.Events061.exact15742RawTerms
def group : MergeGroup := .operator 220147 15742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220147) (leftOrdinal := 0)
    (rightResult := 15742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7211⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7211⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220153

namespace LeftMerge220154
def owner : Owner := ⟨.program ⟨257⟩, ⟨61889⟩⟩
def mergeEvent : Nat := 220154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩] } }
def leftRaw : List Term := Proof.Events859.exact220147RawTerms
def rightRaw : List Term := Proof.Events061.exact15742RawTerms
def group : MergeGroup := .operator 220147 15742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 220147) (leftOrdinal := 1)
    (rightResult := 15742) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220154

namespace LeftMerge220156
def owner : Owner := ⟨.program ⟨257⟩, ⟨61889⟩⟩
def mergeEvent : Nat := 220156
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15735RawTerms
def group : MergeGroup := .relation 220155
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220155) (rhsResult := 15735)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7103⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7103⟩⟩) ⟨7017⟩ 15735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6736⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220156

namespace LeftMerge220170
def owner : Owner := ⟨.program ⟨257⟩, ⟨58907⟩⟩
def mergeEvent : Nat := 220170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58905⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213108RawTerms
def rightRaw : List Term := Proof.Events860.exact220164RawTerms
def group : MergeGroup := .operator 213108 220164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213108) (leftOrdinal := 0)
    (rightResult := 220164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58905⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58905⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge220170

namespace LeftMerge220171
def owner : Owner := ⟨.program ⟨257⟩, ⟨58907⟩⟩
def mergeEvent : Nat := 220171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨56848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58905⟩⟩] } }
def leftRaw : List Term := Proof.Events832.exact213108RawTerms
def rightRaw : List Term := Proof.Events860.exact220164RawTerms
def group : MergeGroup := .operator 213108 220164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 213108) (leftOrdinal := 1)
    (rightResult := 220164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨56848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58905⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨56848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58905⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220171

namespace LeftMerge220173
def owner : Owner := ⟨.program ⟨257⟩, ⟨58907⟩⟩
def mergeEvent : Nat := 220173
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨56848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58120⟩⟩] } }
def rhsRaw : List Term := Proof.Events860.exact220161RawTerms
def group : MergeGroup := .relation 220172
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 220172) (rhsResult := 220161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨56848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58905⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58905⟩⟩) ⟨58120⟩ 220161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58120⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨56848⟩⟩], [⟨.program ⟨257⟩, ⟨58120⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge220173

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
