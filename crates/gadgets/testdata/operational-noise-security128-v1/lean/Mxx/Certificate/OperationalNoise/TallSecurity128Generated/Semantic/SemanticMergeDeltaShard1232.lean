import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge201195
def owner : Owner := ⟨.program ⟨257⟩, ⟨16312⟩⟩
def mergeEvent : Nat := 201195
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events785.exact201189RawTerms
def group : MergeGroup := .operator 192995 201189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 201189) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16309⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201195

namespace LeftMerge201274
def owner : Owner := ⟨.program ⟨257⟩, ⟨15523⟩⟩
def mergeEvent : Nat := 201274
def frameStart : Nat := 201244
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events786.exact201270RawTerms
def rightRaw : List Term := Proof.Events786.exact201267RawTerms
def group : MergeGroup := .operator 201270 201267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201270) (leftOrdinal := 0)
    (rightResult := 201267) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12411⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201274

namespace LeftMerge201304
def owner : Owner := ⟨.program ⟨257⟩, ⟨17136⟩⟩
def mergeEvent : Nat := 201304
def frameStart : Nat := 201244
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events786.exact201300RawTerms
def rightRaw : List Term := Proof.Events786.exact201298RawTerms
def group : MergeGroup := .operator 201300 201298
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201300) (leftOrdinal := 0)
    (rightResult := 201298) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201304

namespace LeftMerge201327
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 201327
def frameStart : Nat := 201244
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events786.exact201323RawTerms
def rightRaw : List Term := Proof.Events786.exact201320RawTerms
def group : MergeGroup := .operator 201323 201320
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201323) (leftOrdinal := 0)
    (rightResult := 201320) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201327

namespace LeftMerge201336
def owner : Owner := ⟨.program ⟨257⟩, ⟨17384⟩⟩
def mergeEvent : Nat := 201336
def frameStart : Nat := 201244
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩] } }
def leftRaw : List Term := Proof.Events786.exact201332RawTerms
def rightRaw : List Term := Proof.Events786.exact201289RawTerms
def group : MergeGroup := .operator 201332 201289
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201332) (leftOrdinal := 0)
    (rightResult := 201289) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17381⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201336

namespace LeftMerge201337
def owner : Owner := ⟨.program ⟨257⟩, ⟨17384⟩⟩
def mergeEvent : Nat := 201337
def frameStart : Nat := 201244
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩] } }
def leftRaw : List Term := Proof.Events786.exact201332RawTerms
def rightRaw : List Term := Proof.Events786.exact201289RawTerms
def group : MergeGroup := .operator 201332 201289
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201332) (leftOrdinal := 1)
    (rightResult := 201289) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17381⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge201337

namespace LeftMerge201339
def owner : Owner := ⟨.program ⟨257⟩, ⟨17384⟩⟩
def mergeEvent : Nat := 201339
def frameStart : Nat := 201244
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16861⟩⟩] } }
def rhsRaw : List Term := Proof.Events786.exact201286RawTerms
def group : MergeGroup := .relation 201338
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 201338) (rhsResult := 201286)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17381⟩⟩) ⟨16861⟩ 201286) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16861⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], [⟨.program ⟨257⟩, ⟨16861⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge201339

namespace LeftMerge201347
def owner : Owner := ⟨.program ⟨257⟩, ⟨15806⟩⟩
def mergeEvent : Nat := 201347
def frameStart : Nat := 201244
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events786.exact201300RawTerms
def rightRaw : List Term := Proof.Events786.exact201343RawTerms
def group : MergeGroup := .operator 201300 201343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201300) (leftOrdinal := 0)
    (rightResult := 201343) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201347

namespace LeftMerge201364
def owner : Owner := ⟨.program ⟨257⟩, ⟨16312⟩⟩
def mergeEvent : Nat := 201364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events786.exact201361RawTerms
def group : MergeGroup := .relation 201363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 201363) (rhsResult := 201361)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 201362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩]⟩) (none) 201361) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201364

namespace LeftMerge201365
def owner : Owner := ⟨.program ⟨257⟩, ⟨16312⟩⟩
def mergeEvent : Nat := 201365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩] } }
def rhsRaw : List Term := Proof.Events786.exact201361RawTerms
def group : MergeGroup := .relation 201363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 201363) (rhsResult := 201361)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 201362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩]⟩) (none) 201361) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge201365

namespace LeftMerge201366
def owner : Owner := ⟨.program ⟨257⟩, ⟨16312⟩⟩
def mergeEvent : Nat := 201366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16861⟩⟩] } }
def rhsRaw : List Term := Proof.Events786.exact201361RawTerms
def group : MergeGroup := .relation 201363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 201363) (rhsResult := 201361)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 201362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩]⟩) (none) 201361) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16861⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], [⟨.program ⟨257⟩, ⟨16861⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201366

namespace LeftMerge201367
def owner : Owner := ⟨.program ⟨257⟩, ⟨16312⟩⟩
def mergeEvent : Nat := 201367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events786.exact201361RawTerms
def group : MergeGroup := .relation 201363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 201363) (rhsResult := 201361)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 201362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16309⟩⟩]⟩) (none) 201361) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge201367

namespace LeftMerge201372
def owner : Owner := ⟨.program ⟨257⟩, ⟨17383⟩⟩
def mergeEvent : Nat := 201372
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16861⟩⟩] } }
def leftRaw : List Term := Proof.Events786.exact201368RawTerms
def rightRaw : List Term := Proof.Events785.exact201182RawTerms
def group : MergeGroup := .operator 201368 201182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201368) (leftOrdinal := 2)
    (rightResult := 201182) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16861⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16861⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨12411⟩⟩, ⟨.program ⟨257⟩, ⟨15522⟩⟩], [⟨.program ⟨257⟩, ⟨16861⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge201372

namespace LeftMerge201373
def owner : Owner := ⟨.program ⟨257⟩, ⟨17383⟩⟩
def mergeEvent : Nat := 201373
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩] } }
def leftRaw : List Term := Proof.Events786.exact201368RawTerms
def rightRaw : List Term := Proof.Events785.exact201182RawTerms
def group : MergeGroup := .operator 201368 201182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201368) (leftOrdinal := 1)
    (rightResult := 201182) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17381⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201373

namespace LeftMerge201381
def owner : Owner := ⟨.program ⟨257⟩, ⟨17819⟩⟩
def mergeEvent : Nat := 201381
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩] } }
def leftRaw : List Term := Proof.Events786.exact201375RawTerms
def rightRaw : List Term := Proof.Events785.exact201098RawTerms
def group : MergeGroup := .operator 201375 201098
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201375) (leftOrdinal := 0)
    (rightResult := 201098) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17817⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge201381

namespace LeftMerge201382
def owner : Owner := ⟨.program ⟨257⟩, ⟨17819⟩⟩
def mergeEvent : Nat := 201382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩] } }
def leftRaw : List Term := Proof.Events786.exact201375RawTerms
def rightRaw : List Term := Proof.Events785.exact201098RawTerms
def group : MergeGroup := .operator 201375 201098
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 201375) (leftOrdinal := 1)
    (rightResult := 201098) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17817⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17817⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge201382

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
