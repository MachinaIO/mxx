import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge239365
def owner : Owner := ⟨.program ⟨257⟩, ⟨34387⟩⟩
def mergeEvent : Nat := 239365
def frameStart : Nat := 239335
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events935.exact239361RawTerms
def rightRaw : List Term := Proof.Events934.exact239358RawTerms
def group : MergeGroup := .operator 239361 239358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239361) (leftOrdinal := 0)
    (rightResult := 239358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13551⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239365

namespace LeftMerge239395
def owner : Owner := ⟨.program ⟨257⟩, ⟨36020⟩⟩
def mergeEvent : Nat := 239395
def frameStart : Nat := 239335
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events935.exact239391RawTerms
def rightRaw : List Term := Proof.Events935.exact239389RawTerms
def group : MergeGroup := .operator 239391 239389
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239391) (leftOrdinal := 0)
    (rightResult := 239389) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239395

namespace LeftMerge239418
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 239418
def frameStart : Nat := 239335
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events935.exact239414RawTerms
def rightRaw : List Term := Proof.Events935.exact239411RawTerms
def group : MergeGroup := .operator 239414 239411
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239414) (leftOrdinal := 0)
    (rightResult := 239411) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239418

namespace LeftMerge239427
def owner : Owner := ⟨.program ⟨257⟩, ⟨36240⟩⟩
def mergeEvent : Nat := 239427
def frameStart : Nat := 239335
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩] } }
def leftRaw : List Term := Proof.Events935.exact239423RawTerms
def rightRaw : List Term := Proof.Events935.exact239380RawTerms
def group : MergeGroup := .operator 239423 239380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239423) (leftOrdinal := 0)
    (rightResult := 239380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36237⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239427

namespace LeftMerge239428
def owner : Owner := ⟨.program ⟨257⟩, ⟨36240⟩⟩
def mergeEvent : Nat := 239428
def frameStart : Nat := 239335
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩] } }
def leftRaw : List Term := Proof.Events935.exact239423RawTerms
def rightRaw : List Term := Proof.Events935.exact239380RawTerms
def group : MergeGroup := .operator 239423 239380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239423) (leftOrdinal := 1)
    (rightResult := 239380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36237⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239428

namespace LeftMerge239430
def owner : Owner := ⟨.program ⟨257⟩, ⟨36240⟩⟩
def mergeEvent : Nat := 239430
def frameStart : Nat := 239335
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35737⟩⟩] } }
def rhsRaw : List Term := Proof.Events935.exact239377RawTerms
def group : MergeGroup := .relation 239429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239429) (rhsResult := 239377)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36237⟩⟩) ⟨35737⟩ 239377) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35737⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], [⟨.program ⟨257⟩, ⟨35737⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239430

namespace LeftMerge239438
def owner : Owner := ⟨.program ⟨257⟩, ⟨34734⟩⟩
def mergeEvent : Nat := 239438
def frameStart : Nat := 239335
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events935.exact239391RawTerms
def rightRaw : List Term := Proof.Events935.exact239434RawTerms
def group : MergeGroup := .operator 239391 239434
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239391) (leftOrdinal := 0)
    (rightResult := 239434) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34732⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239438

namespace LeftMerge239455
def owner : Owner := ⟨.program ⟨257⟩, ⟨35172⟩⟩
def mergeEvent : Nat := 239455
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events935.exact239452RawTerms
def group : MergeGroup := .relation 239454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239454) (rhsResult := 239452)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35169⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 239453 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35169⟩⟩]⟩) (none) 239452) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239455

namespace LeftMerge239456
def owner : Owner := ⟨.program ⟨257⟩, ⟨35172⟩⟩
def mergeEvent : Nat := 239456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩] } }
def rhsRaw : List Term := Proof.Events935.exact239452RawTerms
def group : MergeGroup := .relation 239454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239454) (rhsResult := 239452)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35169⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 239453 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35169⟩⟩]⟩) (none) 239452) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239456

namespace LeftMerge239457
def owner : Owner := ⟨.program ⟨257⟩, ⟨35172⟩⟩
def mergeEvent : Nat := 239457
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35737⟩⟩] } }
def rhsRaw : List Term := Proof.Events935.exact239452RawTerms
def group : MergeGroup := .relation 239454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239454) (rhsResult := 239452)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35169⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 239453 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35169⟩⟩]⟩) (none) 239452) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35737⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], [⟨.program ⟨257⟩, ⟨35737⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239457

namespace LeftMerge239458
def owner : Owner := ⟨.program ⟨257⟩, ⟨35172⟩⟩
def mergeEvent : Nat := 239458
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨34732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events935.exact239452RawTerms
def group : MergeGroup := .relation 239454
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239454) (rhsResult := 239452)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35169⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 239453 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35169⟩⟩]⟩) (none) 239452) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨34732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239458

namespace LeftMerge239463
def owner : Owner := ⟨.program ⟨257⟩, ⟨36239⟩⟩
def mergeEvent : Nat := 239463
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35737⟩⟩] } }
def leftRaw : List Term := Proof.Events935.exact239459RawTerms
def rightRaw : List Term := Proof.Events934.exact239273RawTerms
def group : MergeGroup := .operator 239459 239273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239459) (leftOrdinal := 2)
    (rightResult := 239273) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35737⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35737⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13551⟩⟩, ⟨.program ⟨257⟩, ⟨34386⟩⟩], [⟨.program ⟨257⟩, ⟨35737⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239463

namespace LeftMerge239464
def owner : Owner := ⟨.program ⟨257⟩, ⟨36239⟩⟩
def mergeEvent : Nat := 239464
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩] } }
def leftRaw : List Term := Proof.Events935.exact239459RawTerms
def rightRaw : List Term := Proof.Events934.exact239273RawTerms
def group : MergeGroup := .operator 239459 239273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239459) (leftOrdinal := 1)
    (rightResult := 239273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36237⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239464

namespace LeftMerge239472
def owner : Owner := ⟨.program ⟨257⟩, ⟨36581⟩⟩
def mergeEvent : Nat := 239472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36579⟩⟩] } }
def leftRaw : List Term := Proof.Events935.exact239466RawTerms
def rightRaw : List Term := Proof.Events934.exact239189RawTerms
def group : MergeGroup := .operator 239466 239189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239466) (leftOrdinal := 0)
    (rightResult := 239189) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36579⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36579⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge239472

namespace LeftMerge239473
def owner : Owner := ⟨.program ⟨257⟩, ⟨36581⟩⟩
def mergeEvent : Nat := 239473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨34732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36579⟩⟩] } }
def leftRaw : List Term := Proof.Events935.exact239466RawTerms
def rightRaw : List Term := Proof.Events934.exact239189RawTerms
def group : MergeGroup := .operator 239466 239189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 239466) (leftOrdinal := 1)
    (rightResult := 239189) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨34732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36579⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨34732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36579⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239473

namespace LeftMerge239475
def owner : Owner := ⟨.program ⟨257⟩, ⟨36581⟩⟩
def mergeEvent : Nat := 239475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨34732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35883⟩⟩] } }
def rhsRaw : List Term := Proof.Events934.exact239186RawTerms
def group : MergeGroup := .relation 239474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 239474) (rhsResult := 239186)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨34732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36579⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36579⟩⟩) ⟨35883⟩ 239186) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35883⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨34732⟩⟩], [⟨.program ⟨257⟩, ⟨35883⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge239475

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
