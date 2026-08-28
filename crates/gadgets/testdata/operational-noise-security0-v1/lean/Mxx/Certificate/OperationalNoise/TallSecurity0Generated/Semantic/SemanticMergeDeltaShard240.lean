import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge40108
def owner : Owner := ⟨.program ⟨214⟩, ⟨14758⟩⟩
def mergeEvent : Nat := 40108
def frameStart : Nat := 40048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact40104RawTerms
def rightRaw : List Term := Proof.Events156.exact40102RawTerms
def group : MergeGroup := .operator 40104 40102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40104) (leftOrdinal := 0)
    (rightResult := 40102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40108

namespace LeftMerge40131
def owner : Owner := ⟨.program ⟨214⟩, ⟨7860⟩⟩
def mergeEvent : Nat := 40131
def frameStart : Nat := 40048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact40127RawTerms
def rightRaw : List Term := Proof.Events156.exact40124RawTerms
def group : MergeGroup := .operator 40127 40124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40127) (leftOrdinal := 0)
    (rightResult := 40124) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7858⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40131

namespace LeftMerge40140
def owner : Owner := ⟨.program ⟨214⟩, ⟨26233⟩⟩
def mergeEvent : Nat := 40140
def frameStart : Nat := 40048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact40136RawTerms
def rightRaw : List Term := Proof.Events156.exact40093RawTerms
def group : MergeGroup := .operator 40136 40093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40136) (leftOrdinal := 0)
    (rightResult := 40093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26230⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40140

namespace LeftMerge40141
def owner : Owner := ⟨.program ⟨214⟩, ⟨26233⟩⟩
def mergeEvent : Nat := 40141
def frameStart : Nat := 40048
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact40136RawTerms
def rightRaw : List Term := Proof.Events156.exact40093RawTerms
def group : MergeGroup := .operator 40136 40093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40136) (leftOrdinal := 1)
    (rightResult := 40093) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26230⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40141

namespace LeftMerge40143
def owner : Owner := ⟨.program ⟨214⟩, ⟨26233⟩⟩
def mergeEvent : Nat := 40143
def frameStart : Nat := 40048
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23672⟩⟩] } }
def rhsRaw : List Term := Proof.Events156.exact40090RawTerms
def group : MergeGroup := .relation 40142
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40142) (rhsResult := 40090)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26230⟩⟩) ⟨23672⟩ 40090) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23672⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨23672⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40143

namespace LeftMerge40151
def owner : Owner := ⟨.program ⟨214⟩, ⟨16188⟩⟩
def mergeEvent : Nat := 40151
def frameStart : Nat := 40048
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact40104RawTerms
def rightRaw : List Term := Proof.Events156.exact40147RawTerms
def group : MergeGroup := .operator 40104 40147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40104) (leftOrdinal := 0)
    (rightResult := 40147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40151

namespace LeftMerge40168
def owner : Owner := ⟨.program ⟨214⟩, ⟨19683⟩⟩
def mergeEvent : Nat := 40168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }
def rhsRaw : List Term := Proof.Events156.exact40165RawTerms
def group : MergeGroup := .relation 40167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40167) (rhsResult := 40165)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40166 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩]⟩) (none) 40165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40168

namespace LeftMerge40169
def owner : Owner := ⟨.program ⟨214⟩, ⟨19683⟩⟩
def mergeEvent : Nat := 40169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩] } }
def rhsRaw : List Term := Proof.Events156.exact40165RawTerms
def group : MergeGroup := .relation 40167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40167) (rhsResult := 40165)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40166 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩]⟩) (none) 40165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40169

namespace LeftMerge40170
def owner : Owner := ⟨.program ⟨214⟩, ⟨19683⟩⟩
def mergeEvent : Nat := 40170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23672⟩⟩] } }
def rhsRaw : List Term := Proof.Events156.exact40165RawTerms
def group : MergeGroup := .relation 40167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40167) (rhsResult := 40165)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40166 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩]⟩) (none) 40165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23672⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨23672⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40170

namespace LeftMerge40171
def owner : Owner := ⟨.program ⟨214⟩, ⟨19683⟩⟩
def mergeEvent : Nat := 40171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events156.exact40165RawTerms
def group : MergeGroup := .relation 40167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40167) (rhsResult := 40165)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40166 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19680⟩⟩]⟩) (none) 40165) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40171

namespace LeftMerge40176
def owner : Owner := ⟨.program ⟨214⟩, ⟨26232⟩⟩
def mergeEvent : Nat := 40176
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23672⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact40172RawTerms
def rightRaw : List Term := Proof.Events156.exact39986RawTerms
def group : MergeGroup := .operator 40172 39986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40172) (leftOrdinal := 2)
    (rightResult := 39986) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23672⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23672⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11645⟩⟩, ⟨.program ⟨214⟩, ⟨14659⟩⟩], [⟨.program ⟨214⟩, ⟨23672⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40176

namespace LeftMerge40177
def owner : Owner := ⟨.program ⟨214⟩, ⟨26232⟩⟩
def mergeEvent : Nat := 40177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact40172RawTerms
def rightRaw : List Term := Proof.Events156.exact39986RawTerms
def group : MergeGroup := .operator 40172 39986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40172) (leftOrdinal := 1)
    (rightResult := 39986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40177

namespace LeftMerge40185
def owner : Owner := ⟨.program ⟨214⟩, ⟨28328⟩⟩
def mergeEvent : Nat := 40185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact40179RawTerms
def rightRaw : List Term := Proof.Events155.exact39902RawTerms
def group : MergeGroup := .operator 40179 39902
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40179) (leftOrdinal := 0)
    (rightResult := 39902) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28326⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40185

namespace LeftMerge40186
def owner : Owner := ⟨.program ⟨214⟩, ⟨28328⟩⟩
def mergeEvent : Nat := 40186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩] } }
def leftRaw : List Term := Proof.Events156.exact40179RawTerms
def rightRaw : List Term := Proof.Events155.exact39902RawTerms
def group : MergeGroup := .operator 40179 39902
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40179) (leftOrdinal := 1)
    (rightResult := 39902) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28326⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40186

namespace LeftMerge40188
def owner : Owner := ⟨.program ⟨214⟩, ⟨28328⟩⟩
def mergeEvent : Nat := 40188
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24294⟩⟩] } }
def rhsRaw : List Term := Proof.Events155.exact39899RawTerms
def group : MergeGroup := .relation 40187
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40187) (rhsResult := 39899)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28326⟩⟩) ⟨24294⟩ 39899) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24294⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24294⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40188

namespace LeftMerge40202
def owner : Owner := ⟨.program ⟨214⟩, ⟨21699⟩⟩
def mergeEvent : Nat := 40202
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events157.exact40196RawTerms
def group : MergeGroup := .operator 36137 40196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 40196) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21696⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40202

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
