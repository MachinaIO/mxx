import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge77151
def owner : Owner := ⟨.program ⟨257⟩, ⟨44821⟩⟩
def mergeEvent : Nat := 77151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77145RawTerms
def rightRaw : List Term := Proof.Events300.exact76868RawTerms
def group : MergeGroup := .operator 77145 76868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77145) (leftOrdinal := 0)
    (rightResult := 76868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77151

namespace LeftMerge77152
def owner : Owner := ⟨.program ⟨257⟩, ⟨44821⟩⟩
def mergeEvent : Nat := 77152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77145RawTerms
def rightRaw : List Term := Proof.Events300.exact76868RawTerms
def group : MergeGroup := .operator 77145 76868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77145) (leftOrdinal := 1)
    (rightResult := 76868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77152

namespace LeftMerge77154
def owner : Owner := ⟨.program ⟨257⟩, ⟨44821⟩⟩
def mergeEvent : Nat := 77154
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43995⟩⟩] } }
def rhsRaw : List Term := Proof.Events300.exact76865RawTerms
def group : MergeGroup := .relation 77153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77153) (rhsResult := 76865)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44819⟩⟩) ⟨43995⟩ 76865) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43995⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77154

namespace LeftMerge77168
def owner : Owner := ⟨.program ⟨257⟩, ⟨43659⟩⟩
def mergeEvent : Nat := 77168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events301.exact77162RawTerms
def group : MergeGroup := .operator 75995 77162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 77162) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43656⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77168

namespace LeftMerge77289
def owner : Owner := ⟨.program ⟨257⟩, ⟨44172⟩⟩
def mergeEvent : Nat := 77289
def frameStart : Nat := 77223
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77285RawTerms
def rightRaw : List Term := Proof.Events301.exact77283RawTerms
def group : MergeGroup := .operator 77285 77283
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77285) (leftOrdinal := 0)
    (rightResult := 77283) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77289

namespace LeftMerge77301
def owner : Owner := ⟨.program ⟨257⟩, ⟨44820⟩⟩
def mergeEvent : Nat := 77301
def frameStart : Nat := 77223
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77297RawTerms
def rightRaw : List Term := Proof.Events301.exact77274RawTerms
def group : MergeGroup := .operator 77297 77274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77297) (leftOrdinal := 0)
    (rightResult := 77274) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44819⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77301

namespace LeftMerge77302
def owner : Owner := ⟨.program ⟨257⟩, ⟨44820⟩⟩
def mergeEvent : Nat := 77302
def frameStart : Nat := 77223
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77297RawTerms
def rightRaw : List Term := Proof.Events301.exact77274RawTerms
def group : MergeGroup := .operator 77297 77274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77297) (leftOrdinal := 1)
    (rightResult := 77274) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77302

namespace LeftMerge77304
def owner : Owner := ⟨.program ⟨257⟩, ⟨44820⟩⟩
def mergeEvent : Nat := 77304
def frameStart : Nat := 77223
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43995⟩⟩] } }
def rhsRaw : List Term := Proof.Events301.exact77271RawTerms
def group : MergeGroup := .relation 77303
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77303) (rhsResult := 77271)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44819⟩⟩) ⟨43995⟩ 77271) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43995⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77304

namespace LeftMerge77312
def owner : Owner := ⟨.program ⟨257⟩, ⟨43078⟩⟩
def mergeEvent : Nat := 77312
def frameStart : Nat := 77223
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77285RawTerms
def rightRaw : List Term := Proof.Events301.exact77308RawTerms
def group : MergeGroup := .operator 77285 77308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77285) (leftOrdinal := 0)
    (rightResult := 77308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77312

namespace LeftMerge77329
def owner : Owner := ⟨.program ⟨257⟩, ⟨43659⟩⟩
def mergeEvent : Nat := 77329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }
def rhsRaw : List Term := Proof.Events302.exact77326RawTerms
def group : MergeGroup := .relation 77328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77328) (rhsResult := 77326)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩) (none) 77326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77329

namespace LeftMerge77330
def owner : Owner := ⟨.program ⟨257⟩, ⟨43659⟩⟩
def mergeEvent : Nat := 77330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩] } }
def rhsRaw : List Term := Proof.Events302.exact77326RawTerms
def group : MergeGroup := .relation 77328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77328) (rhsResult := 77326)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩) (none) 77326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77330

namespace LeftMerge77331
def owner : Owner := ⟨.program ⟨257⟩, ⟨43659⟩⟩
def mergeEvent : Nat := 77331
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43995⟩⟩] } }
def rhsRaw : List Term := Proof.Events302.exact77326RawTerms
def group : MergeGroup := .relation 77328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77328) (rhsResult := 77326)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩) (none) 77326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43995⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77331

namespace LeftMerge77332
def owner : Owner := ⟨.program ⟨257⟩, ⟨43659⟩⟩
def mergeEvent : Nat := 77332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events302.exact77326RawTerms
def group : MergeGroup := .relation 77328
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77328) (rhsResult := 77326)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77327 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43656⟩⟩]⟩) (none) 77326) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43077⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43077⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77332

namespace LeftMerge77337
def owner : Owner := ⟨.program ⟨257⟩, ⟨44822⟩⟩
def mergeEvent : Nat := 77337
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩] } }
def leftRaw : List Term := Proof.Events302.exact77333RawTerms
def rightRaw : List Term := Proof.Events301.exact77155RawTerms
def group : MergeGroup := .operator 77333 77155
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77333) (leftOrdinal := 0)
    (rightResult := 77155) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44819⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77337

namespace LeftMerge77338
def owner : Owner := ⟨.program ⟨257⟩, ⟨44822⟩⟩
def mergeEvent : Nat := 77338
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43995⟩⟩] } }
def leftRaw : List Term := Proof.Events302.exact77333RawTerms
def rightRaw : List Term := Proof.Events301.exact77155RawTerms
def group : MergeGroup := .operator 77333 77155
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77333) (leftOrdinal := 2)
    (rightResult := 77155) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43995⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43995⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨43995⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77338

namespace LeftMerge77364
def owner : Owner := ⟨.program ⟨257⟩, ⟨39941⟩⟩
def mergeEvent : Nat := 77364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events012.exact3155RawTerms
def rightRaw : List Term := Proof.Events296.exact75903RawTerms
def group : MergeGroup := .operator 3155 75903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3155) (leftOrdinal := 0)
    (rightResult := 75903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39938⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨39938⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77364

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
