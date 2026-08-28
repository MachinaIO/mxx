import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge174362
def owner : Owner := ⟨.program ⟨257⟩, ⟨47446⟩⟩
def mergeEvent : Nat := 174362
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46656⟩⟩] } }
def leftRaw : List Term := Proof.Events681.exact174357RawTerms
def rightRaw : List Term := Proof.Events680.exact174179RawTerms
def group : MergeGroup := .operator 174357 174179
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 174357) (leftOrdinal := 2)
    (rightResult := 174179) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46656⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46656⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45500⟩⟩], [⟨.program ⟨257⟩, ⟨46656⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge174362

namespace LeftMerge174370
def owner : Owner := ⟨.program ⟨257⟩, ⟨47447⟩⟩
def mergeEvent : Nat := 174370
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }
def leftRaw : List Term := Proof.Events681.exact174364RawTerms
def rightRaw : List Term := Proof.Events060.exact15562RawTerms
def group : MergeGroup := .operator 174364 15562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 174364) (leftOrdinal := 0)
    (rightResult := 15562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7229⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7229⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge174370

namespace LeftMerge174371
def owner : Owner := ⟨.program ⟨257⟩, ⟨47447⟩⟩
def mergeEvent : Nat := 174371
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩] } }
def leftRaw : List Term := Proof.Events681.exact174364RawTerms
def rightRaw : List Term := Proof.Events060.exact15562RawTerms
def group : MergeGroup := .operator 174364 15562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 174364) (leftOrdinal := 1)
    (rightResult := 15562) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge174371

namespace LeftMerge174373
def owner : Owner := ⟨.program ⟨257⟩, ⟨47447⟩⟩
def mergeEvent : Nat := 174373
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15555RawTerms
def group : MergeGroup := .relation 174372
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 174372) (rhsResult := 15555)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7151⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7151⟩⟩) ⟨7041⟩ 15555) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45731⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge174373

namespace LeftMerge174387
def owner : Owner := ⟨.program ⟨257⟩, ⟨44765⟩⟩
def mergeEvent : Nat := 174387
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩] } }
def leftRaw : List Term := Proof.Events644.exact164895RawTerms
def rightRaw : List Term := Proof.Events681.exact174381RawTerms
def group : MergeGroup := .operator 164895 174381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164895) (leftOrdinal := 0)
    (rightResult := 174381) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44763⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge174387

namespace LeftMerge174388
def owner : Owner := ⟨.program ⟨257⟩, ⟨44765⟩⟩
def mergeEvent : Nat := 174388
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩] } }
def leftRaw : List Term := Proof.Events644.exact164895RawTerms
def rightRaw : List Term := Proof.Events681.exact174381RawTerms
def group : MergeGroup := .operator 164895 174381
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164895) (leftOrdinal := 1)
    (rightResult := 174381) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44763⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge174388

namespace LeftMerge174390
def owner : Owner := ⟨.program ⟨257⟩, ⟨44765⟩⟩
def mergeEvent : Nat := 174390
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43976⟩⟩] } }
def rhsRaw : List Term := Proof.Events681.exact174378RawTerms
def group : MergeGroup := .relation 174389
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 174389) (rhsResult := 174378)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44763⟩⟩) ⟨43976⟩ 174378) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43976⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨43976⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge174390

namespace LeftMerge174404
def owner : Owner := ⟨.program ⟨257⟩, ⟨43615⟩⟩
def mergeEvent : Nat := 174404
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43612⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events681.exact174398RawTerms
def group : MergeGroup := .operator 163745 174398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 174398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43612⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge174404

namespace LeftMerge174525
def owner : Owner := ⟨.program ⟨257⟩, ⟨44164⟩⟩
def mergeEvent : Nat := 174525
def frameStart : Nat := 174459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events681.exact174521RawTerms
def rightRaw : List Term := Proof.Events681.exact174519RawTerms
def group : MergeGroup := .operator 174521 174519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 174521) (leftOrdinal := 0)
    (rightResult := 174519) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge174525

namespace LeftMerge174537
def owner : Owner := ⟨.program ⟨257⟩, ⟨44764⟩⟩
def mergeEvent : Nat := 174537
def frameStart : Nat := 174459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩] } }
def leftRaw : List Term := Proof.Events681.exact174533RawTerms
def rightRaw : List Term := Proof.Events681.exact174510RawTerms
def group : MergeGroup := .operator 174533 174510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 174533) (leftOrdinal := 0)
    (rightResult := 174510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44763⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge174537

namespace LeftMerge174538
def owner : Owner := ⟨.program ⟨257⟩, ⟨44764⟩⟩
def mergeEvent : Nat := 174538
def frameStart : Nat := 174459
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩] } }
def leftRaw : List Term := Proof.Events681.exact174533RawTerms
def rightRaw : List Term := Proof.Events681.exact174510RawTerms
def group : MergeGroup := .operator 174533 174510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 174533) (leftOrdinal := 1)
    (rightResult := 174510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44763⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge174538

namespace LeftMerge174540
def owner : Owner := ⟨.program ⟨257⟩, ⟨44764⟩⟩
def mergeEvent : Nat := 174540
def frameStart : Nat := 174459
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43976⟩⟩] } }
def rhsRaw : List Term := Proof.Events681.exact174507RawTerms
def group : MergeGroup := .relation 174539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 174539) (rhsResult := 174507)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44763⟩⟩) ⟨43976⟩ 174507) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43976⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨43976⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge174540

namespace LeftMerge174548
def owner : Owner := ⟨.program ⟨257⟩, ⟨43056⟩⟩
def mergeEvent : Nat := 174548
def frameStart : Nat := 174459
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43054⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events681.exact174521RawTerms
def rightRaw : List Term := Proof.Events681.exact174544RawTerms
def group : MergeGroup := .operator 174521 174544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 174521) (leftOrdinal := 0)
    (rightResult := 174544) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43054⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨43054⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge174548

namespace LeftMerge174565
def owner : Owner := ⟨.program ⟨257⟩, ⟨43615⟩⟩
def mergeEvent : Nat := 174565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩] } }
def rhsRaw : List Term := Proof.Events681.exact174562RawTerms
def group : MergeGroup := .relation 174564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 174564) (rhsResult := 174562)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43612⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 174563 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43612⟩⟩]⟩) (none) 174562) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7227⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7227⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge174565

namespace LeftMerge174566
def owner : Owner := ⟨.program ⟨257⟩, ⟨43615⟩⟩
def mergeEvent : Nat := 174566
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩] } }
def rhsRaw : List Term := Proof.Events681.exact174562RawTerms
def group : MergeGroup := .relation 174564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 174564) (rhsResult := 174562)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43612⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 174563 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43612⟩⟩]⟩) (none) 174562) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44763⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge174566

namespace LeftMerge174567
def owner : Owner := ⟨.program ⟨257⟩, ⟨43615⟩⟩
def mergeEvent : Nat := 174567
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43976⟩⟩] } }
def rhsRaw : List Term := Proof.Events681.exact174562RawTerms
def group : MergeGroup := .relation 174564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 174564) (rhsResult := 174562)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43612⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 174563 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43612⟩⟩]⟩) (none) 174562) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43976⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨43976⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge174567

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
