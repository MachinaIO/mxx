import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge164202
def owner : Owner := ⟨.program ⟨257⟩, ⟨14845⟩⟩
def mergeEvent : Nat := 164202
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def rhsRaw : List Term := Proof.Events068.exact17581RawTerms
def group : MergeGroup := .relation 164201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164201) (rhsResult := 17581)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9562⟩⟩) ⟨7284⟩ 17581) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164202

namespace LeftMerge164203
def owner : Owner := ⟨.program ⟨257⟩, ⟨14845⟩⟩
def mergeEvent : Nat := 164203
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164194RawTerms
def rightRaw : List Term := Proof.Events068.exact17611RawTerms
def group : MergeGroup := .operator 164194 17611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164194) (leftOrdinal := 0)
    (rightResult := 17611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164203

namespace LeftMerge164208
def owner : Owner := ⟨.program ⟨257⟩, ⟨45257⟩⟩
def mergeEvent : Nat := 164208
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164204RawTerms
def rightRaw : List Term := Proof.Events641.exact164174RawTerms
def group : MergeGroup := .operator 164204 164174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164204) (leftOrdinal := 1)
    (rightResult := 164174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164208

namespace LeftMerge164216
def owner : Owner := ⟨.program ⟨257⟩, ⟨47024⟩⟩
def mergeEvent : Nat := 164216
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164210RawTerms
def rightRaw : List Term := Proof.Events641.exact164146RawTerms
def group : MergeGroup := .operator 164210 164146
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164210) (leftOrdinal := 1)
    (rightResult := 164146) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47023⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164216

namespace LeftMerge164218
def owner : Owner := ⟨.program ⟨257⟩, ⟨47024⟩⟩
def mergeEvent : Nat := 164218
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46493⟩⟩] } }
def rhsRaw : List Term := Proof.Events641.exact164143RawTerms
def group : MergeGroup := .relation 164217
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164217) (rhsResult := 164143)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47023⟩⟩) ⟨46493⟩ 164143) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46493⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨46493⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164218

namespace LeftMerge164219
def owner : Owner := ⟨.program ⟨257⟩, ⟨47024⟩⟩
def mergeEvent : Nat := 164219
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164210RawTerms
def rightRaw : List Term := Proof.Events641.exact164146RawTerms
def group : MergeGroup := .operator 164210 164146
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164210) (leftOrdinal := 0)
    (rightResult := 164146) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47023⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164219

namespace LeftMerge164233
def owner : Owner := ⟨.program ⟨257⟩, ⟨45952⟩⟩
def mergeEvent : Nat := 164233
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events641.exact164227RawTerms
def group : MergeGroup := .operator 163745 164227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 164227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45949⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164233

namespace LeftMerge164312
def owner : Owner := ⟨.program ⟨257⟩, ⟨45251⟩⟩
def mergeEvent : Nat := 164312
def frameStart : Nat := 164282
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events641.exact164308RawTerms
def rightRaw : List Term := Proof.Events641.exact164305RawTerms
def group : MergeGroup := .operator 164308 164305
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164308) (leftOrdinal := 0)
    (rightResult := 164305) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164312

namespace LeftMerge164342
def owner : Owner := ⟨.program ⟨257⟩, ⟨46764⟩⟩
def mergeEvent : Nat := 164342
def frameStart : Nat := 164282
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164338RawTerms
def rightRaw : List Term := Proof.Events641.exact164336RawTerms
def group : MergeGroup := .operator 164338 164336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164338) (leftOrdinal := 0)
    (rightResult := 164336) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164342

namespace LeftMerge164365
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 164365
def frameStart : Nat := 164282
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164361RawTerms
def rightRaw : List Term := Proof.Events642.exact164358RawTerms
def group : MergeGroup := .operator 164361 164358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164361) (leftOrdinal := 0)
    (rightResult := 164358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164365

namespace LeftMerge164374
def owner : Owner := ⟨.program ⟨257⟩, ⟨47026⟩⟩
def mergeEvent : Nat := 164374
def frameStart : Nat := 164282
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164370RawTerms
def rightRaw : List Term := Proof.Events641.exact164327RawTerms
def group : MergeGroup := .operator 164370 164327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164370) (leftOrdinal := 0)
    (rightResult := 164327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47023⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164374

namespace LeftMerge164375
def owner : Owner := ⟨.program ⟨257⟩, ⟨47026⟩⟩
def mergeEvent : Nat := 164375
def frameStart : Nat := 164282
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩] } }
def leftRaw : List Term := Proof.Events642.exact164370RawTerms
def rightRaw : List Term := Proof.Events641.exact164327RawTerms
def group : MergeGroup := .operator 164370 164327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164370) (leftOrdinal := 1)
    (rightResult := 164327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47023⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164375

namespace LeftMerge164377
def owner : Owner := ⟨.program ⟨257⟩, ⟨47026⟩⟩
def mergeEvent : Nat := 164377
def frameStart : Nat := 164282
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46493⟩⟩] } }
def rhsRaw : List Term := Proof.Events641.exact164324RawTerms
def group : MergeGroup := .relation 164376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164376) (rhsResult := 164324)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47023⟩⟩) ⟨46493⟩ 164324) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46493⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨46493⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164377

namespace LeftMerge164385
def owner : Owner := ⟨.program ⟨257⟩, ⟨45502⟩⟩
def mergeEvent : Nat := 164385
def frameStart : Nat := 164282
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164338RawTerms
def rightRaw : List Term := Proof.Events642.exact164381RawTerms
def group : MergeGroup := .operator 164338 164381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164338) (leftOrdinal := 0)
    (rightResult := 164381) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164385

namespace LeftMerge164402
def owner : Owner := ⟨.program ⟨257⟩, ⟨45952⟩⟩
def mergeEvent : Nat := 164402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events642.exact164399RawTerms
def group : MergeGroup := .relation 164401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164401) (rhsResult := 164399)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164400 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩) (none) 164399) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164402

namespace LeftMerge164403
def owner : Owner := ⟨.program ⟨257⟩, ⟨45952⟩⟩
def mergeEvent : Nat := 164403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩] } }
def rhsRaw : List Term := Proof.Events642.exact164399RawTerms
def group : MergeGroup := .relation 164401
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164401) (rhsResult := 164399)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164400 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩) (none) 164399) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47023⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164403

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
