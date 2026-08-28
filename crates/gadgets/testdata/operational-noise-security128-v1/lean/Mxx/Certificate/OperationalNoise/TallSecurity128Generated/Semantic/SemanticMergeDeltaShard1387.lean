import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge225284
def owner : Owner := ⟨.program ⟨257⟩, ⟨30591⟩⟩
def mergeEvent : Nat := 225284
def frameStart : Nat := 225192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩] } }
def leftRaw : List Term := Proof.Events880.exact225280RawTerms
def rightRaw : List Term := Proof.Events879.exact225237RawTerms
def group : MergeGroup := .operator 225280 225237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 225280) (leftOrdinal := 0)
    (rightResult := 225237) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30588⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge225284

namespace LeftMerge225285
def owner : Owner := ⟨.program ⟨257⟩, ⟨30591⟩⟩
def mergeEvent : Nat := 225285
def frameStart : Nat := 225192
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩] } }
def leftRaw : List Term := Proof.Events880.exact225280RawTerms
def rightRaw : List Term := Proof.Events879.exact225237RawTerms
def group : MergeGroup := .operator 225280 225237
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 225280) (leftOrdinal := 1)
    (rightResult := 225237) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30588⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge225285

namespace LeftMerge225287
def owner : Owner := ⟨.program ⟨257⟩, ⟨30591⟩⟩
def mergeEvent : Nat := 225287
def frameStart : Nat := 225192
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30083⟩⟩] } }
def rhsRaw : List Term := Proof.Events879.exact225234RawTerms
def group : MergeGroup := .relation 225286
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 225286) (rhsResult := 225234)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30588⟩⟩) ⟨30083⟩ 225234) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30083⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], [⟨.program ⟨257⟩, ⟨30083⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge225287

namespace LeftMerge225295
def owner : Owner := ⟨.program ⟨257⟩, ⟨29082⟩⟩
def mergeEvent : Nat := 225295
def frameStart : Nat := 225192
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events879.exact225248RawTerms
def rightRaw : List Term := Proof.Events880.exact225291RawTerms
def group : MergeGroup := .operator 225248 225291
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 225248) (leftOrdinal := 0)
    (rightResult := 225291) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29080⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge225295

namespace LeftMerge225312
def owner : Owner := ⟨.program ⟨257⟩, ⟨29522⟩⟩
def mergeEvent : Nat := 225312
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events880.exact225309RawTerms
def group : MergeGroup := .relation 225311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 225311) (rhsResult := 225309)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 225310 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29519⟩⟩]⟩) (none) 225309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge225312

namespace LeftMerge225313
def owner : Owner := ⟨.program ⟨257⟩, ⟨29522⟩⟩
def mergeEvent : Nat := 225313
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩] } }
def rhsRaw : List Term := Proof.Events880.exact225309RawTerms
def group : MergeGroup := .relation 225311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 225311) (rhsResult := 225309)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 225310 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29519⟩⟩]⟩) (none) 225309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge225313

namespace LeftMerge225314
def owner : Owner := ⟨.program ⟨257⟩, ⟨29522⟩⟩
def mergeEvent : Nat := 225314
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30083⟩⟩] } }
def rhsRaw : List Term := Proof.Events880.exact225309RawTerms
def group : MergeGroup := .relation 225311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 225311) (rhsResult := 225309)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 225310 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29519⟩⟩]⟩) (none) 225309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30083⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], [⟨.program ⟨257⟩, ⟨30083⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge225314

namespace LeftMerge225315
def owner : Owner := ⟨.program ⟨257⟩, ⟨29522⟩⟩
def mergeEvent : Nat := 225315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events880.exact225309RawTerms
def group : MergeGroup := .relation 225311
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 225311) (rhsResult := 225309)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 225310 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29519⟩⟩]⟩) (none) 225309) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge225315

namespace LeftMerge225320
def owner : Owner := ⟨.program ⟨257⟩, ⟨30590⟩⟩
def mergeEvent : Nat := 225320
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30083⟩⟩] } }
def leftRaw : List Term := Proof.Events880.exact225316RawTerms
def rightRaw : List Term := Proof.Events879.exact225130RawTerms
def group : MergeGroup := .operator 225316 225130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 225316) (leftOrdinal := 2)
    (rightResult := 225130) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30083⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30083⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨13266⟩⟩, ⟨.program ⟨257⟩, ⟨28750⟩⟩], [⟨.program ⟨257⟩, ⟨30083⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge225320

namespace LeftMerge225321
def owner : Owner := ⟨.program ⟨257⟩, ⟨30590⟩⟩
def mergeEvent : Nat := 225321
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩] } }
def leftRaw : List Term := Proof.Events880.exact225316RawTerms
def rightRaw : List Term := Proof.Events879.exact225130RawTerms
def group : MergeGroup := .operator 225316 225130
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 225316) (leftOrdinal := 1)
    (rightResult := 225130) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30588⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge225321

namespace LeftMerge225329
def owner : Owner := ⟨.program ⟨257⟩, ⟨30946⟩⟩
def mergeEvent : Nat := 225329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30944⟩⟩] } }
def leftRaw : List Term := Proof.Events880.exact225323RawTerms
def rightRaw : List Term := Proof.Events879.exact225046RawTerms
def group : MergeGroup := .operator 225323 225046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 225323) (leftOrdinal := 0)
    (rightResult := 225046) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30944⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30944⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge225329

namespace LeftMerge225330
def owner : Owner := ⟨.program ⟨257⟩, ⟨30946⟩⟩
def mergeEvent : Nat := 225330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30944⟩⟩] } }
def leftRaw : List Term := Proof.Events880.exact225323RawTerms
def rightRaw : List Term := Proof.Events879.exact225046RawTerms
def group : MergeGroup := .operator 225323 225046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 225323) (leftOrdinal := 1)
    (rightResult := 225046) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30944⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30944⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge225330

namespace LeftMerge225332
def owner : Owner := ⟨.program ⟨257⟩, ⟨30946⟩⟩
def mergeEvent : Nat := 225332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30232⟩⟩] } }
def rhsRaw : List Term := Proof.Events879.exact225043RawTerms
def group : MergeGroup := .relation 225331
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 225331) (rhsResult := 225043)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30944⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30944⟩⟩) ⟨30232⟩ 225043) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30232⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨29080⟩⟩], [⟨.program ⟨257⟩, ⟨30232⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge225332

namespace LeftMerge225346
def owner : Owner := ⟨.program ⟨257⟩, ⟨29819⟩⟩
def mergeEvent : Nat := 225346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29816⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events880.exact225340RawTerms
def group : MergeGroup := .operator 222245 225340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 225340) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29816⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29816⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge225346

namespace LeftMerge225467
def owner : Owner := ⟨.program ⟨257⟩, ⟨30444⟩⟩
def mergeEvent : Nat := 225467
def frameStart : Nat := 225401
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events880.exact225463RawTerms
def rightRaw : List Term := Proof.Events880.exact225461RawTerms
def group : MergeGroup := .operator 225463 225461
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 225463) (leftOrdinal := 0)
    (rightResult := 225461) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29080⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge225467

namespace LeftMerge225479
def owner : Owner := ⟨.program ⟨257⟩, ⟨30945⟩⟩
def mergeEvent : Nat := 225479
def frameStart : Nat := 225401
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30944⟩⟩] } }
def leftRaw : List Term := Proof.Events880.exact225475RawTerms
def rightRaw : List Term := Proof.Events880.exact225452RawTerms
def group : MergeGroup := .operator 225475 225452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 225475) (leftOrdinal := 0)
    (rightResult := 225452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30944⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30944⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge225479

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
