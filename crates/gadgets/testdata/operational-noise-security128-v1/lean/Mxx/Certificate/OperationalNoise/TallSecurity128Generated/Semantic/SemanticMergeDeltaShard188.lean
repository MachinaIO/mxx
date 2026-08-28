import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge34133
def owner : Owner := ⟨.program ⟨257⟩, ⟨37331⟩⟩
def mergeEvent : Nat := 34133
def frameStart : Nat := 34103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events133.exact34129RawTerms
def rightRaw : List Term := Proof.Events133.exact34126RawTerms
def group : MergeGroup := .operator 34129 34126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34129) (leftOrdinal := 0)
    (rightResult := 34126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14016⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34133

namespace LeftMerge34163
def owner : Owner := ⟨.program ⟨257⟩, ⟨38744⟩⟩
def mergeEvent : Nat := 34163
def frameStart : Nat := 34103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34159RawTerms
def rightRaw : List Term := Proof.Events133.exact34157RawTerms
def group : MergeGroup := .operator 34159 34157
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34159) (leftOrdinal := 0)
    (rightResult := 34157) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34163

namespace LeftMerge34186
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 34186
def frameStart : Nat := 34103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34182RawTerms
def rightRaw : List Term := Proof.Events133.exact34179RawTerms
def group : MergeGroup := .operator 34182 34179
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34182) (leftOrdinal := 0)
    (rightResult := 34179) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34186

namespace LeftMerge34195
def owner : Owner := ⟨.program ⟨257⟩, ⟨39041⟩⟩
def mergeEvent : Nat := 34195
def frameStart : Nat := 34103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34191RawTerms
def rightRaw : List Term := Proof.Events133.exact34148RawTerms
def group : MergeGroup := .operator 34191 34148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34191) (leftOrdinal := 0)
    (rightResult := 34148) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39038⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34195

namespace LeftMerge34196
def owner : Owner := ⟨.program ⟨257⟩, ⟨39041⟩⟩
def mergeEvent : Nat := 34196
def frameStart : Nat := 34103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34191RawTerms
def rightRaw : List Term := Proof.Events133.exact34148RawTerms
def group : MergeGroup := .operator 34191 34148
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34191) (leftOrdinal := 1)
    (rightResult := 34148) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39038⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34196

namespace LeftMerge34198
def owner : Owner := ⟨.program ⟨257⟩, ⟨39041⟩⟩
def mergeEvent : Nat := 34198
def frameStart : Nat := 34103
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38483⟩⟩] } }
def rhsRaw : List Term := Proof.Events133.exact34145RawTerms
def group : MergeGroup := .relation 34197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34197) (rhsResult := 34145)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39038⟩⟩) ⟨38483⟩ 34145) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38483⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], [⟨.program ⟨257⟩, ⟨38483⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34198

namespace LeftMerge34206
def owner : Owner := ⟨.program ⟨257⟩, ⟨37502⟩⟩
def mergeEvent : Nat := 34206
def frameStart : Nat := 34103
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34159RawTerms
def rightRaw : List Term := Proof.Events133.exact34202RawTerms
def group : MergeGroup := .operator 34159 34202
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34159) (leftOrdinal := 0)
    (rightResult := 34202) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34206

namespace LeftMerge34223
def owner : Owner := ⟨.program ⟨257⟩, ⟨37962⟩⟩
def mergeEvent : Nat := 34223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events133.exact34220RawTerms
def group : MergeGroup := .relation 34222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34222) (rhsResult := 34220)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37959⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 34221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37959⟩⟩]⟩) (none) 34220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34223

namespace LeftMerge34224
def owner : Owner := ⟨.program ⟨257⟩, ⟨37962⟩⟩
def mergeEvent : Nat := 34224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩] } }
def rhsRaw : List Term := Proof.Events133.exact34220RawTerms
def group : MergeGroup := .relation 34222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34222) (rhsResult := 34220)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37959⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 34221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37959⟩⟩]⟩) (none) 34220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34224

namespace LeftMerge34225
def owner : Owner := ⟨.program ⟨257⟩, ⟨37962⟩⟩
def mergeEvent : Nat := 34225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38483⟩⟩] } }
def rhsRaw : List Term := Proof.Events133.exact34220RawTerms
def group : MergeGroup := .relation 34222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34222) (rhsResult := 34220)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37959⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 34221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37959⟩⟩]⟩) (none) 34220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38483⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], [⟨.program ⟨257⟩, ⟨38483⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34225

namespace LeftMerge34226
def owner : Owner := ⟨.program ⟨257⟩, ⟨37962⟩⟩
def mergeEvent : Nat := 34226
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events133.exact34220RawTerms
def group : MergeGroup := .relation 34222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34222) (rhsResult := 34220)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37959⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 34221 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37959⟩⟩]⟩) (none) 34220) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34226

namespace LeftMerge34231
def owner : Owner := ⟨.program ⟨257⟩, ⟨39040⟩⟩
def mergeEvent : Nat := 34231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38483⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34227RawTerms
def rightRaw : List Term := Proof.Events132.exact34041RawTerms
def group : MergeGroup := .operator 34227 34041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34227) (leftOrdinal := 2)
    (rightResult := 34041) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38483⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38483⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14016⟩⟩, ⟨.program ⟨257⟩, ⟨37330⟩⟩], [⟨.program ⟨257⟩, ⟨38483⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34231

namespace LeftMerge34232
def owner : Owner := ⟨.program ⟨257⟩, ⟨39040⟩⟩
def mergeEvent : Nat := 34232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34227RawTerms
def rightRaw : List Term := Proof.Events132.exact34041RawTerms
def group : MergeGroup := .operator 34227 34041
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34227) (leftOrdinal := 1)
    (rightResult := 34041) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨39038⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34232

namespace LeftMerge34240
def owner : Owner := ⟨.program ⟨257⟩, ⟨39536⟩⟩
def mergeEvent : Nat := 34240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39534⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34234RawTerms
def rightRaw : List Term := Proof.Events132.exact33957RawTerms
def group : MergeGroup := .operator 34234 33957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34234) (leftOrdinal := 0)
    (rightResult := 33957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39534⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34240

namespace LeftMerge34241
def owner : Owner := ⟨.program ⟨257⟩, ⟨39536⟩⟩
def mergeEvent : Nat := 34241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39534⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34234RawTerms
def rightRaw : List Term := Proof.Events132.exact33957RawTerms
def group : MergeGroup := .operator 34234 33957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34234) (leftOrdinal := 1)
    (rightResult := 33957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39534⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39534⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34241

namespace LeftMerge34243
def owner : Owner := ⟨.program ⟨257⟩, ⟨39536⟩⟩
def mergeEvent : Nat := 34243
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38662⟩⟩] } }
def rhsRaw : List Term := Proof.Events132.exact33954RawTerms
def group : MergeGroup := .relation 34242
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34242) (rhsResult := 33954)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39534⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39534⟩⟩) ⟨38662⟩ 33954) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38662⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38662⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34243

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
