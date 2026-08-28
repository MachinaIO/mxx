import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge272337
def owner : Owner := ⟨.program ⟨257⟩, ⟨50324⟩⟩
def mergeEvent : Nat := 272337
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events051.exact13112RawTerms
def rightRaw : List Term := Proof.Events1039.exact266028RawTerms
def group : MergeGroup := .operator 13112 266028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13112) (leftOrdinal := 0)
    (rightResult := 266028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272337

namespace LeftMerge272342
def owner : Owner := ⟨.program ⟨257⟩, ⟨7644⟩⟩
def mergeEvent : Nat := 272342
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }
def leftRaw : List Term := Proof.Events1038.exact265898RawTerms
def rightRaw : List Term := Proof.Events092.exact23634RawTerms
def group : MergeGroup := .operator 265898 23634
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 265898) (leftOrdinal := 0)
    (rightResult := 23634) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272342

namespace LeftMerge272359
def owner : Owner := ⟨.program ⟨257⟩, ⟨50327⟩⟩
def mergeEvent : Nat := 272359
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272353RawTerms
def rightRaw : List Term := Proof.Events092.exact23623RawTerms
def group : MergeGroup := .operator 272353 23623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272353) (leftOrdinal := 1)
    (rightResult := 23623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272359

namespace LeftMerge272361
def owner : Owner := ⟨.program ⟨257⟩, ⟨50327⟩⟩
def mergeEvent : Nat := 272361
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }
def rhsRaw : List Term := Proof.Events092.exact23593RawTerms
def group : MergeGroup := .relation 272360
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272360) (rhsResult := 23593)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9580⟩⟩) ⟨7308⟩ 23593) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272361

namespace LeftMerge272362
def owner : Owner := ⟨.program ⟨257⟩, ⟨50327⟩⟩
def mergeEvent : Nat := 272362
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272353RawTerms
def rightRaw : List Term := Proof.Events092.exact23623RawTerms
def group : MergeGroup := .operator 272353 23623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272353) (leftOrdinal := 0)
    (rightResult := 23623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272362

namespace LeftMerge272367
def owner : Owner := ⟨.program ⟨257⟩, ⟨50328⟩⟩
def mergeEvent : Nat := 272367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272363RawTerms
def rightRaw : List Term := Proof.Events1063.exact272333RawTerms
def group : MergeGroup := .operator 272363 272333
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272363) (leftOrdinal := 1)
    (rightResult := 272333) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7308⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨7308⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272367

namespace LeftMerge272375
def owner : Owner := ⟨.program ⟨257⟩, ⟨52429⟩⟩
def mergeEvent : Nat := 272375
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272369RawTerms
def rightRaw : List Term := Proof.Events1063.exact272305RawTerms
def group : MergeGroup := .operator 272369 272305
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272369) (leftOrdinal := 1)
    (rightResult := 272305) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52428⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272375

namespace LeftMerge272377
def owner : Owner := ⟨.program ⟨257⟩, ⟨52429⟩⟩
def mergeEvent : Nat := 272377
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51959⟩⟩] } }
def rhsRaw : List Term := Proof.Events1063.exact272302RawTerms
def group : MergeGroup := .relation 272376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272376) (rhsResult := 272302)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52428⟩⟩) ⟨51959⟩ 272302) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨51959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272377

namespace LeftMerge272378
def owner : Owner := ⟨.program ⟨257⟩, ⟨52429⟩⟩
def mergeEvent : Nat := 272378
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩] } }
def leftRaw : List Term := Proof.Events1063.exact272369RawTerms
def rightRaw : List Term := Proof.Events1063.exact272305RawTerms
def group : MergeGroup := .operator 272369 272305
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272369) (leftOrdinal := 0)
    (rightResult := 272305) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52428⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272378

namespace LeftMerge272392
def owner : Owner := ⟨.program ⟨257⟩, ⟨51369⟩⟩
def mergeEvent : Nat := 272392
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1064.exact272386RawTerms
def group : MergeGroup := .operator 266120 272386
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 272386) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51366⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51366⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272392

namespace LeftMerge272471
def owner : Owner := ⟨.program ⟨257⟩, ⟨50321⟩⟩
def mergeEvent : Nat := 272471
def frameStart : Nat := 272441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1064.exact272467RawTerms
def rightRaw : List Term := Proof.Events1064.exact272464RawTerms
def group : MergeGroup := .operator 272467 272464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272467) (leftOrdinal := 0)
    (rightResult := 272464) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24430⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272471

namespace LeftMerge272501
def owner : Owner := ⟨.program ⟨257⟩, ⟨52256⟩⟩
def mergeEvent : Nat := 272501
def frameStart : Nat := 272441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1064.exact272497RawTerms
def rightRaw : List Term := Proof.Events1064.exact272495RawTerms
def group : MergeGroup := .operator 272497 272495
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272497) (leftOrdinal := 0)
    (rightResult := 272495) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272501

namespace LeftMerge272524
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 272524
def frameStart : Nat := 272441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events1064.exact272520RawTerms
def rightRaw : List Term := Proof.Events1064.exact272517RawTerms
def group : MergeGroup := .operator 272520 272517
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272520) (leftOrdinal := 0)
    (rightResult := 272517) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272524

namespace LeftMerge272533
def owner : Owner := ⟨.program ⟨257⟩, ⟨52431⟩⟩
def mergeEvent : Nat := 272533
def frameStart : Nat := 272441
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩] } }
def leftRaw : List Term := Proof.Events1064.exact272529RawTerms
def rightRaw : List Term := Proof.Events1064.exact272486RawTerms
def group : MergeGroup := .operator 272529 272486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272529) (leftOrdinal := 0)
    (rightResult := 272486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52428⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272533

namespace LeftMerge272534
def owner : Owner := ⟨.program ⟨257⟩, ⟨52431⟩⟩
def mergeEvent : Nat := 272534
def frameStart : Nat := 272441
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩] } }
def leftRaw : List Term := Proof.Events1064.exact272529RawTerms
def rightRaw : List Term := Proof.Events1064.exact272486RawTerms
def group : MergeGroup := .operator 272529 272486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272529) (leftOrdinal := 1)
    (rightResult := 272486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52428⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272534

namespace LeftMerge272536
def owner : Owner := ⟨.program ⟨257⟩, ⟨52431⟩⟩
def mergeEvent : Nat := 272536
def frameStart : Nat := 272441
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51959⟩⟩] } }
def rhsRaw : List Term := Proof.Events1064.exact272483RawTerms
def group : MergeGroup := .relation 272535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272535) (rhsResult := 272483)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52428⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52428⟩⟩) ⟨51959⟩ 272483) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24430⟩⟩, ⟨.program ⟨257⟩, ⟨50320⟩⟩], [⟨.program ⟨257⟩, ⟨51959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272536

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
