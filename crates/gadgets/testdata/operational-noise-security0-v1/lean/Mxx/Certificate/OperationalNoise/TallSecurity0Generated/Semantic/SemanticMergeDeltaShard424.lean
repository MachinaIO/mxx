import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge69418
def owner : Owner := ⟨.program ⟨214⟩, ⟨19671⟩⟩
def mergeEvent : Nat := 69418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }
def rhsRaw : List Term := Proof.Events271.exact69415RawTerms
def group : MergeGroup := .relation 69417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69417) (rhsResult := 69415)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69416 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩) (none) 69415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69418

namespace LeftMerge69419
def owner : Owner := ⟨.program ⟨214⟩, ⟨19671⟩⟩
def mergeEvent : Nat := 69419
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩] } }
def rhsRaw : List Term := Proof.Events271.exact69415RawTerms
def group : MergeGroup := .relation 69417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69417) (rhsResult := 69415)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69416 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩) (none) 69415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69419

namespace LeftMerge69420
def owner : Owner := ⟨.program ⟨214⟩, ⟨19671⟩⟩
def mergeEvent : Nat := 69420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23666⟩⟩] } }
def rhsRaw : List Term := Proof.Events271.exact69415RawTerms
def group : MergeGroup := .relation 69417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69417) (rhsResult := 69415)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69416 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩) (none) 69415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23666⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69420

namespace LeftMerge69421
def owner : Owner := ⟨.program ⟨214⟩, ⟨19671⟩⟩
def mergeEvent : Nat := 69421
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events271.exact69415RawTerms
def group : MergeGroup := .relation 69417
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69417) (rhsResult := 69415)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69416 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19668⟩⟩]⟩) (none) 69415) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69421

namespace LeftMerge69426
def owner : Owner := ⟨.program ⟨214⟩, ⟨26217⟩⟩
def mergeEvent : Nat := 69426
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23666⟩⟩] } }
def leftRaw : List Term := Proof.Events271.exact69422RawTerms
def rightRaw : List Term := Proof.Events270.exact69236RawTerms
def group : MergeGroup := .operator 69422 69236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69422) (leftOrdinal := 2)
    (rightResult := 69236) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23666⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23666⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨11633⟩⟩, ⟨.program ⟨214⟩, ⟨14632⟩⟩], [⟨.program ⟨214⟩, ⟨23666⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69426

namespace LeftMerge69427
def owner : Owner := ⟨.program ⟨214⟩, ⟨26217⟩⟩
def mergeEvent : Nat := 69427
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩] } }
def leftRaw : List Term := Proof.Events271.exact69422RawTerms
def rightRaw : List Term := Proof.Events270.exact69236RawTerms
def group : MergeGroup := .operator 69422 69236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69422) (leftOrdinal := 1)
    (rightResult := 69236) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6762⟩⟩, ⟨.program ⟨214⟩, ⟨7858⟩⟩, ⟨.program ⟨214⟩, ⟨26215⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69427

namespace LeftMerge69435
def owner : Owner := ⟨.program ⟨214⟩, ⟨28289⟩⟩
def mergeEvent : Nat := 69435
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩] } }
def leftRaw : List Term := Proof.Events271.exact69429RawTerms
def rightRaw : List Term := Proof.Events270.exact69152RawTerms
def group : MergeGroup := .operator 69429 69152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69429) (leftOrdinal := 0)
    (rightResult := 69152) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28287⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69435

namespace LeftMerge69436
def owner : Owner := ⟨.program ⟨214⟩, ⟨28289⟩⟩
def mergeEvent : Nat := 69436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩] } }
def leftRaw : List Term := Proof.Events271.exact69429RawTerms
def rightRaw : List Term := Proof.Events270.exact69152RawTerms
def group : MergeGroup := .operator 69429 69152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69429) (leftOrdinal := 1)
    (rightResult := 69152) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28287⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69436

namespace LeftMerge69438
def owner : Owner := ⟨.program ⟨214⟩, ⟨28289⟩⟩
def mergeEvent : Nat := 69438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24285⟩⟩] } }
def rhsRaw : List Term := Proof.Events270.exact69149RawTerms
def group : MergeGroup := .relation 69437
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69437) (rhsResult := 69149)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28287⟩⟩) ⟨24285⟩ 69149) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24285⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69438

namespace LeftMerge69452
def owner : Owner := ⟨.program ⟨214⟩, ⟨21687⟩⟩
def mergeEvent : Nat := 69452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩] } }
def leftRaw : List Term := Proof.Events255.exact65387RawTerms
def rightRaw : List Term := Proof.Events271.exact69446RawTerms
def group : MergeGroup := .operator 65387 69446
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 65387) (leftOrdinal := 0)
    (rightResult := 69446) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21684⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69452

namespace LeftMerge69573
def owner : Owner := ⟨.program ⟨214⟩, ⟨16216⟩⟩
def mergeEvent : Nat := 69573
def frameStart : Nat := 69507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events271.exact69569RawTerms
def rightRaw : List Term := Proof.Events271.exact69567RawTerms
def group : MergeGroup := .operator 69569 69567
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69569) (leftOrdinal := 0)
    (rightResult := 69567) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69573

namespace LeftMerge69585
def owner : Owner := ⟨.program ⟨214⟩, ⟨28288⟩⟩
def mergeEvent : Nat := 69585
def frameStart : Nat := 69507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩] } }
def leftRaw : List Term := Proof.Events271.exact69581RawTerms
def rightRaw : List Term := Proof.Events271.exact69558RawTerms
def group : MergeGroup := .operator 69581 69558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69581) (leftOrdinal := 0)
    (rightResult := 69558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28287⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69585

namespace LeftMerge69586
def owner : Owner := ⟨.program ⟨214⟩, ⟨28288⟩⟩
def mergeEvent : Nat := 69586
def frameStart : Nat := 69507
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩] } }
def leftRaw : List Term := Proof.Events271.exact69581RawTerms
def rightRaw : List Term := Proof.Events271.exact69558RawTerms
def group : MergeGroup := .operator 69581 69558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69581) (leftOrdinal := 1)
    (rightResult := 69558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28287⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69586

namespace LeftMerge69588
def owner : Owner := ⟨.program ⟨214⟩, ⟨28288⟩⟩
def mergeEvent : Nat := 69588
def frameStart : Nat := 69507
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16174⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24285⟩⟩] } }
def rhsRaw : List Term := Proof.Events271.exact69555RawTerms
def group : MergeGroup := .relation 69587
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69587) (rhsResult := 69555)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28287⟩⟩) ⟨24285⟩ 69555) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24285⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16174⟩⟩], [⟨.program ⟨214⟩, ⟨24285⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69588

namespace LeftMerge69596
def owner : Owner := ⟨.program ⟨214⟩, ⟨18338⟩⟩
def mergeEvent : Nat := 69596
def frameStart : Nat := 69507
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events271.exact69569RawTerms
def rightRaw : List Term := Proof.Events271.exact69592RawTerms
def group : MergeGroup := .operator 69569 69592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69569) (leftOrdinal := 0)
    (rightResult := 69592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69596

namespace LeftMerge69613
def owner : Owner := ⟨.program ⟨214⟩, ⟨21687⟩⟩
def mergeEvent : Nat := 69613
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }
def rhsRaw : List Term := Proof.Events271.exact69610RawTerms
def group : MergeGroup := .relation 69612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69612) (rhsResult := 69610)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69611 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21684⟩⟩]⟩) (none) 69610) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69613

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
