import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge90096
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 14) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90096

namespace LeftMerge90097
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90097
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 13) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90097

namespace LeftMerge90098
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90098
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 12) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90098

namespace LeftMerge90099
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 11) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90099

namespace LeftMerge90100
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90100
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 10) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90100

namespace LeftMerge90101
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90101
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 9) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90101

namespace LeftMerge90102
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90102
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 8) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90102

namespace LeftMerge90103
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90103
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 7) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90103

namespace LeftMerge90104
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90104
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 6) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90104

namespace LeftMerge90105
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90105
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 5) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90105

namespace LeftMerge90106
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 4) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90106

namespace LeftMerge90107
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90107

namespace LeftMerge90108
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90108
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90108

namespace LeftMerge90109
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90109
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90109

namespace LeftMerge90110
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90110
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90110

namespace LeftMerge90111
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90111
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 34) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18170⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18170⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90111

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
