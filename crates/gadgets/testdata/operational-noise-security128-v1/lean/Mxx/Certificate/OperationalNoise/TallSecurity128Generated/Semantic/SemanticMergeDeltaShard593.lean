import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge99326
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99326
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 7)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99326

namespace LeftMerge99327
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99327
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 33)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99327

namespace LeftMerge99329
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60196⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events353.exact90500RawTerms
def group : MergeGroup := .relation 99328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99328) (rhsResult := 90500)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60196⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 90500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨60196⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99329

namespace LeftMerge99330
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 6)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99330

namespace LeftMerge99331
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 32)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57216⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99331

namespace LeftMerge99333
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99333
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57216⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events353.exact90500RawTerms
def group : MergeGroup := .relation 99332
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99332) (rhsResult := 90500)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57216⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 90500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨57216⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99333

namespace LeftMerge99334
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99334
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 5)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99334

namespace LeftMerge99335
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99335
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 31)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99335

namespace LeftMerge99337
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99337
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54236⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events353.exact90500RawTerms
def group : MergeGroup := .relation 99336
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99336) (rhsResult := 90500)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 90500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨54236⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99337

namespace LeftMerge99338
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99338
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 4)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99338

namespace LeftMerge99339
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99339
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 30)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99339

namespace LeftMerge99341
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99341
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51256⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events353.exact90500RawTerms
def group : MergeGroup := .relation 99340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99340) (rhsResult := 90500)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 90500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51256⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99341

namespace LeftMerge99342
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99342
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 3)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99342

namespace LeftMerge99343
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99343
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 23)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99343

namespace LeftMerge99345
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99345
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32201⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }
def rhsRaw : List Term := Proof.Events353.exact90500RawTerms
def group : MergeGroup := .relation 99344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99344) (rhsResult := 90500)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32201⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71405⟩⟩) ⟨68860⟩ 90500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68860⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨32201⟩⟩], [⟨.program ⟨257⟩, ⟨68860⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99345

namespace LeftMerge99346
def owner : Owner := ⟨.program ⟨257⟩, ⟨71407⟩⟩
def mergeEvent : Nat := 99346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99280RawTerms
def rightRaw : List Term := Proof.Events353.exact90503RawTerms
def group : MergeGroup := .operator 99280 90503
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99280) (leftOrdinal := 2)
    (rightResult := 90503) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71405⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71405⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99346

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
