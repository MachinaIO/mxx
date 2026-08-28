import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge253116
def owner : Owner := ⟨.program ⟨257⟩, ⟨40502⟩⟩
def mergeEvent : Nat := 253116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events988.exact253113RawTerms
def group : MergeGroup := .relation 253115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253115) (rhsResult := 253113)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 253114 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40499⟩⟩]⟩) (none) 253113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253116

namespace LeftMerge253117
def owner : Owner := ⟨.program ⟨257⟩, ⟨40502⟩⟩
def mergeEvent : Nat := 253117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41564⟩⟩] } }
def rhsRaw : List Term := Proof.Events988.exact253113RawTerms
def group : MergeGroup := .relation 253115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253115) (rhsResult := 253113)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 253114 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40499⟩⟩]⟩) (none) 253113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41564⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41564⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253117

namespace LeftMerge253118
def owner : Owner := ⟨.program ⟨257⟩, ⟨40502⟩⟩
def mergeEvent : Nat := 253118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14106⟩⟩, ⟨.program ⟨257⟩, ⟨39674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41079⟩⟩] } }
def rhsRaw : List Term := Proof.Events988.exact253113RawTerms
def group : MergeGroup := .relation 253115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253115) (rhsResult := 253113)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 253114 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40499⟩⟩]⟩) (none) 253113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14106⟩⟩, ⟨.program ⟨257⟩, ⟨39674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41079⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14106⟩⟩, ⟨.program ⟨257⟩, ⟨39674⟩⟩], [⟨.program ⟨257⟩, ⟨41079⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253118

namespace LeftMerge253119
def owner : Owner := ⟨.program ⟨257⟩, ⟨40502⟩⟩
def mergeEvent : Nat := 253119
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events988.exact253113RawTerms
def group : MergeGroup := .relation 253115
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253115) (rhsResult := 253113)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 253114 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40499⟩⟩]⟩) (none) 253113) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253119

namespace LeftMerge253124
def owner : Owner := ⟨.program ⟨257⟩, ⟨41566⟩⟩
def mergeEvent : Nat := 253124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14106⟩⟩, ⟨.program ⟨257⟩, ⟨39674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41079⟩⟩] } }
def leftRaw : List Term := Proof.Events988.exact253120RawTerms
def rightRaw : List Term := Proof.Events988.exact252934RawTerms
def group : MergeGroup := .operator 253120 252934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253120) (leftOrdinal := 2)
    (rightResult := 252934) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14106⟩⟩, ⟨.program ⟨257⟩, ⟨39674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41079⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14106⟩⟩, ⟨.program ⟨257⟩, ⟨39674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41079⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨14106⟩⟩, ⟨.program ⟨257⟩, ⟨39674⟩⟩], [⟨.program ⟨257⟩, ⟨41079⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253124

namespace LeftMerge253125
def owner : Owner := ⟨.program ⟨257⟩, ⟨41566⟩⟩
def mergeEvent : Nat := 253125
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41564⟩⟩] } }
def leftRaw : List Term := Proof.Events988.exact253120RawTerms
def rightRaw : List Term := Proof.Events988.exact252934RawTerms
def group : MergeGroup := .operator 253120 252934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253120) (leftOrdinal := 1)
    (rightResult := 252934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41564⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41564⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41564⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253125

namespace LeftMerge253133
def owner : Owner := ⟨.program ⟨257⟩, ⟨41866⟩⟩
def mergeEvent : Nat := 253133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩] } }
def leftRaw : List Term := Proof.Events988.exact253127RawTerms
def rightRaw : List Term := Proof.Events987.exact252850RawTerms
def group : MergeGroup := .operator 253127 252850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253127) (leftOrdinal := 0)
    (rightResult := 252850) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41864⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253133

namespace LeftMerge253134
def owner : Owner := ⟨.program ⟨257⟩, ⟨41866⟩⟩
def mergeEvent : Nat := 253134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩] } }
def leftRaw : List Term := Proof.Events988.exact253127RawTerms
def rightRaw : List Term := Proof.Events987.exact252850RawTerms
def group : MergeGroup := .operator 253127 252850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253127) (leftOrdinal := 1)
    (rightResult := 252850) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41864⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253134

namespace LeftMerge253136
def owner : Owner := ⟨.program ⟨257⟩, ⟨41866⟩⟩
def mergeEvent : Nat := 253136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41216⟩⟩] } }
def rhsRaw : List Term := Proof.Events987.exact252847RawTerms
def group : MergeGroup := .relation 253135
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253135) (rhsResult := 252847)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41864⟩⟩) ⟨41216⟩ 252847) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41216⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨40068⟩⟩], [⟨.program ⟨257⟩, ⟨41216⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253136

namespace LeftMerge253150
def owner : Owner := ⟨.program ⟨257⟩, ⟨40759⟩⟩
def mergeEvent : Nat := 253150
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40756⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events988.exact253144RawTerms
def group : MergeGroup := .operator 251495 253144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 253144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40756⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253150

namespace LeftMerge253271
def owner : Owner := ⟨.program ⟨257⟩, ⟨41448⟩⟩
def mergeEvent : Nat := 253271
def frameStart : Nat := 253205
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events989.exact253267RawTerms
def rightRaw : List Term := Proof.Events989.exact253265RawTerms
def group : MergeGroup := .operator 253267 253265
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253267) (leftOrdinal := 0)
    (rightResult := 253265) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253271

namespace LeftMerge253283
def owner : Owner := ⟨.program ⟨257⟩, ⟨41865⟩⟩
def mergeEvent : Nat := 253283
def frameStart : Nat := 253205
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩] } }
def leftRaw : List Term := Proof.Events989.exact253279RawTerms
def rightRaw : List Term := Proof.Events989.exact253256RawTerms
def group : MergeGroup := .operator 253279 253256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253279) (leftOrdinal := 0)
    (rightResult := 253256) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41864⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253283

namespace LeftMerge253284
def owner : Owner := ⟨.program ⟨257⟩, ⟨41865⟩⟩
def mergeEvent : Nat := 253284
def frameStart : Nat := 253205
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩] } }
def leftRaw : List Term := Proof.Events989.exact253279RawTerms
def rightRaw : List Term := Proof.Events989.exact253256RawTerms
def group : MergeGroup := .operator 253279 253256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253279) (leftOrdinal := 1)
    (rightResult := 253256) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41864⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253284

namespace LeftMerge253286
def owner : Owner := ⟨.program ⟨257⟩, ⟨41865⟩⟩
def mergeEvent : Nat := 253286
def frameStart : Nat := 253205
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40068⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41216⟩⟩] } }
def rhsRaw : List Term := Proof.Events989.exact253253RawTerms
def group : MergeGroup := .relation 253285
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253285) (rhsResult := 253253)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40068⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41864⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41864⟩⟩) ⟨41216⟩ 253253) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41216⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40068⟩⟩], [⟨.program ⟨257⟩, ⟨41216⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge253286

namespace LeftMerge253294
def owner : Owner := ⟨.program ⟨257⟩, ⟨40255⟩⟩
def mergeEvent : Nat := 253294
def frameStart : Nat := 253205
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events989.exact253267RawTerms
def rightRaw : List Term := Proof.Events989.exact253290RawTerms
def group : MergeGroup := .operator 253267 253290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 253267) (leftOrdinal := 0)
    (rightResult := 253290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253294

namespace LeftMerge253311
def owner : Owner := ⟨.program ⟨257⟩, ⟨40759⟩⟩
def mergeEvent : Nat := 253311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }
def rhsRaw : List Term := Proof.Events989.exact253308RawTerms
def group : MergeGroup := .relation 253310
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 253310) (rhsResult := 253308)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 253309 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40756⟩⟩]⟩) (none) 253308) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge253311

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
