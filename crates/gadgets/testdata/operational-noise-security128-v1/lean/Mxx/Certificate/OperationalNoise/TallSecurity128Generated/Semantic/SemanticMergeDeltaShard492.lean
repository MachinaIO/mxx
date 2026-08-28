import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge83206
def owner : Owner := ⟨.program ⟨257⟩, ⟨21645⟩⟩
def mergeEvent : Nat := 83206
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83202RawTerms
def rightRaw : List Term := Proof.Events324.exact83172RawTerms
def group : MergeGroup := .operator 83202 83172
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83202) (leftOrdinal := 1)
    (rightResult := 83172) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83206

namespace LeftMerge83214
def owner : Owner := ⟨.program ⟨257⟩, ⟨23506⟩⟩
def mergeEvent : Nat := 83214
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83208RawTerms
def rightRaw : List Term := Proof.Events324.exact83144RawTerms
def group : MergeGroup := .operator 83208 83144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83208) (leftOrdinal := 1)
    (rightResult := 83144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23505⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83214

namespace LeftMerge83216
def owner : Owner := ⟨.program ⟨257⟩, ⟨23506⟩⟩
def mergeEvent : Nat := 83216
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22965⟩⟩] } }
def rhsRaw : List Term := Proof.Events324.exact83141RawTerms
def group : MergeGroup := .relation 83215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83215) (rhsResult := 83141)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23505⟩⟩) ⟨22965⟩ 83141) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22965⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83216

namespace LeftMerge83217
def owner : Owner := ⟨.program ⟨257⟩, ⟨23506⟩⟩
def mergeEvent : Nat := 83217
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83208RawTerms
def rightRaw : List Term := Proof.Events324.exact83144RawTerms
def group : MergeGroup := .operator 83208 83144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83208) (leftOrdinal := 0)
    (rightResult := 83144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23505⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83217

namespace LeftMerge83231
def owner : Owner := ⟨.program ⟨257⟩, ⟨22432⟩⟩
def mergeEvent : Nat := 83231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events325.exact83225RawTerms
def group : MergeGroup := .operator 75995 83225
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 83225) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22429⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83231

namespace LeftMerge83310
def owner : Owner := ⟨.program ⟨257⟩, ⟨21639⟩⟩
def mergeEvent : Nat := 83310
def frameStart : Nat := 83280
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events325.exact83306RawTerms
def rightRaw : List Term := Proof.Events325.exact83303RawTerms
def group : MergeGroup := .operator 83306 83303
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83306) (leftOrdinal := 0)
    (rightResult := 83303) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21191⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83310

namespace LeftMerge83340
def owner : Owner := ⟨.program ⟨257⟩, ⟨23232⟩⟩
def mergeEvent : Nat := 83340
def frameStart : Nat := 83280
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83336RawTerms
def rightRaw : List Term := Proof.Events325.exact83334RawTerms
def group : MergeGroup := .operator 83336 83334
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83336) (leftOrdinal := 0)
    (rightResult := 83334) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83340

namespace LeftMerge83363
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 83363
def frameStart : Nat := 83280
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83359RawTerms
def rightRaw : List Term := Proof.Events325.exact83356RawTerms
def group : MergeGroup := .operator 83359 83356
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83359) (leftOrdinal := 0)
    (rightResult := 83356) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83363

namespace LeftMerge83372
def owner : Owner := ⟨.program ⟨257⟩, ⟨23508⟩⟩
def mergeEvent : Nat := 83372
def frameStart : Nat := 83280
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83368RawTerms
def rightRaw : List Term := Proof.Events325.exact83325RawTerms
def group : MergeGroup := .operator 83368 83325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83368) (leftOrdinal := 0)
    (rightResult := 83325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23505⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83372

namespace LeftMerge83373
def owner : Owner := ⟨.program ⟨257⟩, ⟨23508⟩⟩
def mergeEvent : Nat := 83373
def frameStart : Nat := 83280
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83368RawTerms
def rightRaw : List Term := Proof.Events325.exact83325RawTerms
def group : MergeGroup := .operator 83368 83325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83368) (leftOrdinal := 1)
    (rightResult := 83325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23505⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83373

namespace LeftMerge83375
def owner : Owner := ⟨.program ⟨257⟩, ⟨23508⟩⟩
def mergeEvent : Nat := 83375
def frameStart : Nat := 83280
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22965⟩⟩] } }
def rhsRaw : List Term := Proof.Events325.exact83322RawTerms
def group : MergeGroup := .relation 83374
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83374) (rhsResult := 83322)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23505⟩⟩) ⟨22965⟩ 83322) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22965⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83375

namespace LeftMerge83383
def owner : Owner := ⟨.program ⟨257⟩, ⟨21858⟩⟩
def mergeEvent : Nat := 83383
def frameStart : Nat := 83280
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83336RawTerms
def rightRaw : List Term := Proof.Events325.exact83379RawTerms
def group : MergeGroup := .operator 83336 83379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83336) (leftOrdinal := 0)
    (rightResult := 83379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83383

namespace LeftMerge83400
def owner : Owner := ⟨.program ⟨257⟩, ⟨22432⟩⟩
def mergeEvent : Nat := 83400
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events325.exact83397RawTerms
def group : MergeGroup := .relation 83399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83399) (rhsResult := 83397)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩) (none) 83397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83400

namespace LeftMerge83401
def owner : Owner := ⟨.program ⟨257⟩, ⟨22432⟩⟩
def mergeEvent : Nat := 83401
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩] } }
def rhsRaw : List Term := Proof.Events325.exact83397RawTerms
def group : MergeGroup := .relation 83399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83399) (rhsResult := 83397)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩) (none) 83397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83401

namespace LeftMerge83402
def owner : Owner := ⟨.program ⟨257⟩, ⟨22432⟩⟩
def mergeEvent : Nat := 83402
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22965⟩⟩] } }
def rhsRaw : List Term := Proof.Events325.exact83397RawTerms
def group : MergeGroup := .relation 83399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83399) (rhsResult := 83397)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩) (none) 83397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22965⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83402

namespace LeftMerge83403
def owner : Owner := ⟨.program ⟨257⟩, ⟨22432⟩⟩
def mergeEvent : Nat := 83403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events325.exact83397RawTerms
def group : MergeGroup := .relation 83399
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83399) (rhsResult := 83397)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83398 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22429⟩⟩]⟩) (none) 83397) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83403

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
