import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge90112
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90112
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 30) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17085⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17085⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90112

namespace LeftMerge90113
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90113
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 29) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16798⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16798⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90113

namespace LeftMerge90114
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90114
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 28) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16679⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16679⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90114

namespace LeftMerge90115
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 35) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90115

namespace LeftMerge90116
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 33) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90116

namespace LeftMerge90117
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 31) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90117

namespace LeftMerge90118
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 27) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90118

namespace LeftMerge90119
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90119
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 36) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90119

namespace LeftMerge90120
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90120
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 26) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90120

namespace LeftMerge90121
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90121
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 25) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90121

namespace LeftMerge90122
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90122
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 24) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90122

namespace LeftMerge90123
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90123
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 23) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90123

namespace LeftMerge90124
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 22) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15629⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15629⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90124

namespace LeftMerge90125
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90125
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 32) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17327⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17327⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90125

namespace LeftMerge90126
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90126
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 21) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15366⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90126

namespace LeftMerge90127
def owner : Owner := ⟨.program ⟨214⟩, ⟨18562⟩⟩
def mergeEvent : Nat := 90127
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events351.exact90089RawTerms
def group : MergeGroup := .relation 90091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90091) (rhsResult := 90089)
    (sourceTermOrdinal := 20) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 90090 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18559⟩⟩]⟩) (none) 90089) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15310⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15310⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90127

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
