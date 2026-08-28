import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge76950
def owner : Owner := ⟨.program ⟨257⟩, ⟨44366⟩⟩
def mergeEvent : Nat := 76950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43825⟩⟩] } }
def rhsRaw : List Term := Proof.Events300.exact76875RawTerms
def group : MergeGroup := .relation 76949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 76949) (rhsResult := 76875)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44365⟩⟩) ⟨43825⟩ 76875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43825⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge76950

namespace LeftMerge76951
def owner : Owner := ⟨.program ⟨257⟩, ⟨44366⟩⟩
def mergeEvent : Nat := 76951
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩] } }
def leftRaw : List Term := Proof.Events300.exact76942RawTerms
def rightRaw : List Term := Proof.Events300.exact76878RawTerms
def group : MergeGroup := .operator 76942 76878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 76942) (leftOrdinal := 0)
    (rightResult := 76878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76951

namespace LeftMerge76965
def owner : Owner := ⟨.program ⟨257⟩, ⟨43292⟩⟩
def mergeEvent : Nat := 76965
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events300.exact76959RawTerms
def group : MergeGroup := .operator 75995 76959
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 76959) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43289⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge76965

namespace LeftMerge77044
def owner : Owner := ⟨.program ⟨257⟩, ⟨42619⟩⟩
def mergeEvent : Nat := 77044
def frameStart : Nat := 77014
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events300.exact77040RawTerms
def rightRaw : List Term := Proof.Events300.exact77037RawTerms
def group : MergeGroup := .operator 77040 77037
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77040) (leftOrdinal := 0)
    (rightResult := 77037) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14571⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77044

namespace LeftMerge77074
def owner : Owner := ⟨.program ⟨257⟩, ⟨44092⟩⟩
def mergeEvent : Nat := 77074
def frameStart : Nat := 77014
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77070RawTerms
def rightRaw : List Term := Proof.Events301.exact77068RawTerms
def group : MergeGroup := .operator 77070 77068
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77070) (leftOrdinal := 0)
    (rightResult := 77068) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77074

namespace LeftMerge77097
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def mergeEvent : Nat := 77097
def frameStart : Nat := 77014
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77093RawTerms
def rightRaw : List Term := Proof.Events301.exact77090RawTerms
def group : MergeGroup := .operator 77093 77090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77093) (leftOrdinal := 0)
    (rightResult := 77090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77097

namespace LeftMerge77106
def owner : Owner := ⟨.program ⟨257⟩, ⟨44368⟩⟩
def mergeEvent : Nat := 77106
def frameStart : Nat := 77014
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77102RawTerms
def rightRaw : List Term := Proof.Events301.exact77059RawTerms
def group : MergeGroup := .operator 77102 77059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77102) (leftOrdinal := 0)
    (rightResult := 77059) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44365⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77106

namespace LeftMerge77107
def owner : Owner := ⟨.program ⟨257⟩, ⟨44368⟩⟩
def mergeEvent : Nat := 77107
def frameStart : Nat := 77014
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77102RawTerms
def rightRaw : List Term := Proof.Events301.exact77059RawTerms
def group : MergeGroup := .operator 77102 77059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77102) (leftOrdinal := 1)
    (rightResult := 77059) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77107

namespace LeftMerge77109
def owner : Owner := ⟨.program ⟨257⟩, ⟨44368⟩⟩
def mergeEvent : Nat := 77109
def frameStart : Nat := 77014
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43825⟩⟩] } }
def rhsRaw : List Term := Proof.Events301.exact77056RawTerms
def group : MergeGroup := .relation 77108
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77108) (rhsResult := 77056)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44365⟩⟩) ⟨43825⟩ 77056) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43825⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77109

namespace LeftMerge77117
def owner : Owner := ⟨.program ⟨257⟩, ⟨42838⟩⟩
def mergeEvent : Nat := 77117
def frameStart : Nat := 77014
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77070RawTerms
def rightRaw : List Term := Proof.Events301.exact77113RawTerms
def group : MergeGroup := .operator 77070 77113
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77070) (leftOrdinal := 0)
    (rightResult := 77113) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77117

namespace LeftMerge77134
def owner : Owner := ⟨.program ⟨257⟩, ⟨43292⟩⟩
def mergeEvent : Nat := 77134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events301.exact77131RawTerms
def group : MergeGroup := .relation 77133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77133) (rhsResult := 77131)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77132 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩) (none) 77131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77134

namespace LeftMerge77135
def owner : Owner := ⟨.program ⟨257⟩, ⟨43292⟩⟩
def mergeEvent : Nat := 77135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩] } }
def rhsRaw : List Term := Proof.Events301.exact77131RawTerms
def group : MergeGroup := .relation 77133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77133) (rhsResult := 77131)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77132 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩) (none) 77131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77135

namespace LeftMerge77136
def owner : Owner := ⟨.program ⟨257⟩, ⟨43292⟩⟩
def mergeEvent : Nat := 77136
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43825⟩⟩] } }
def rhsRaw : List Term := Proof.Events301.exact77131RawTerms
def group : MergeGroup := .relation 77133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77133) (rhsResult := 77131)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77132 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩) (none) 77131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43825⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77136

namespace LeftMerge77137
def owner : Owner := ⟨.program ⟨257⟩, ⟨43292⟩⟩
def mergeEvent : Nat := 77137
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events301.exact77131RawTerms
def group : MergeGroup := .relation 77133
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77133) (rhsResult := 77131)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77132 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43289⟩⟩]⟩) (none) 77131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42836⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨42836⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77137

namespace LeftMerge77142
def owner : Owner := ⟨.program ⟨257⟩, ⟨44367⟩⟩
def mergeEvent : Nat := 77142
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43825⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77138RawTerms
def rightRaw : List Term := Proof.Events300.exact76952RawTerms
def group : MergeGroup := .operator 77138 76952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77138) (leftOrdinal := 2)
    (rightResult := 76952) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43825⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43825⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨14571⟩⟩, ⟨.program ⟨257⟩, ⟨42618⟩⟩], [⟨.program ⟨257⟩, ⟨43825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77142

namespace LeftMerge77143
def owner : Owner := ⟨.program ⟨257⟩, ⟨44367⟩⟩
def mergeEvent : Nat := 77143
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩] } }
def leftRaw : List Term := Proof.Events301.exact77138RawTerms
def rightRaw : List Term := Proof.Events300.exact76952RawTerms
def group : MergeGroup := .operator 77138 76952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77138) (leftOrdinal := 1)
    (rightResult := 76952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44365⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77143

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
