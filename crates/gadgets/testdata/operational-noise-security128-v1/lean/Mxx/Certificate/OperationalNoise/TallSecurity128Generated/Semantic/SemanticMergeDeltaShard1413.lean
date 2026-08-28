import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge229963
def owner : Owner := ⟨.program ⟨257⟩, ⟨19142⟩⟩
def mergeEvent : Nat := 229963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events898.exact229957RawTerms
def group : MergeGroup := .operator 222245 229957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 229957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge229963

namespace LeftMerge230042
def owner : Owner := ⟨.program ⟨257⟩, ⟨18251⟩⟩
def mergeEvent : Nat := 230042
def frameStart : Nat := 230012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events898.exact230038RawTerms
def rightRaw : List Term := Proof.Events898.exact230035RawTerms
def group : MergeGroup := .operator 230038 230035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230038) (leftOrdinal := 0)
    (rightResult := 230035) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12666⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230042

namespace LeftMerge230072
def owner : Owner := ⟨.program ⟨257⟩, ⟨19984⟩⟩
def mergeEvent : Nat := 230072
def frameStart : Nat := 230012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230068RawTerms
def rightRaw : List Term := Proof.Events898.exact230066RawTerms
def group : MergeGroup := .operator 230068 230066
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230068) (leftOrdinal := 0)
    (rightResult := 230066) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230072

namespace LeftMerge230095
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def mergeEvent : Nat := 230095
def frameStart : Nat := 230012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230091RawTerms
def rightRaw : List Term := Proof.Events898.exact230088RawTerms
def group : MergeGroup := .operator 230091 230088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230091) (leftOrdinal := 0)
    (rightResult := 230088) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230095

namespace LeftMerge230104
def owner : Owner := ⟨.program ⟨257⟩, ⟨20211⟩⟩
def mergeEvent : Nat := 230104
def frameStart : Nat := 230012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230100RawTerms
def rightRaw : List Term := Proof.Events898.exact230057RawTerms
def group : MergeGroup := .operator 230100 230057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230100) (leftOrdinal := 0)
    (rightResult := 230057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20208⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230104

namespace LeftMerge230105
def owner : Owner := ⟨.program ⟨257⟩, ⟨20211⟩⟩
def mergeEvent : Nat := 230105
def frameStart : Nat := 230012
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230100RawTerms
def rightRaw : List Term := Proof.Events898.exact230057RawTerms
def group : MergeGroup := .operator 230100 230057
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230100) (leftOrdinal := 1)
    (rightResult := 230057) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20208⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230105

namespace LeftMerge230107
def owner : Owner := ⟨.program ⟨257⟩, ⟨20211⟩⟩
def mergeEvent : Nat := 230107
def frameStart : Nat := 230012
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19703⟩⟩] } }
def rhsRaw : List Term := Proof.Events898.exact230054RawTerms
def group : MergeGroup := .relation 230106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230106) (rhsResult := 230054)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20208⟩⟩) ⟨19703⟩ 230054) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19703⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], [⟨.program ⟨257⟩, ⟨19703⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230107

namespace LeftMerge230115
def owner : Owner := ⟨.program ⟨257⟩, ⟨18582⟩⟩
def mergeEvent : Nat := 230115
def frameStart : Nat := 230012
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230068RawTerms
def rightRaw : List Term := Proof.Events898.exact230111RawTerms
def group : MergeGroup := .operator 230068 230111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230068) (leftOrdinal := 0)
    (rightResult := 230111) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230115

namespace LeftMerge230132
def owner : Owner := ⟨.program ⟨257⟩, ⟨19142⟩⟩
def mergeEvent : Nat := 230132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events898.exact230129RawTerms
def group : MergeGroup := .relation 230131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230131) (rhsResult := 230129)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230130 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩]⟩) (none) 230129) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230132

namespace LeftMerge230133
def owner : Owner := ⟨.program ⟨257⟩, ⟨19142⟩⟩
def mergeEvent : Nat := 230133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩] } }
def rhsRaw : List Term := Proof.Events898.exact230129RawTerms
def group : MergeGroup := .relation 230131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230131) (rhsResult := 230129)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230130 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩]⟩) (none) 230129) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230133

namespace LeftMerge230134
def owner : Owner := ⟨.program ⟨257⟩, ⟨19142⟩⟩
def mergeEvent : Nat := 230134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19703⟩⟩] } }
def rhsRaw : List Term := Proof.Events898.exact230129RawTerms
def group : MergeGroup := .relation 230131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230131) (rhsResult := 230129)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230130 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩]⟩) (none) 230129) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19703⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], [⟨.program ⟨257⟩, ⟨19703⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230134

namespace LeftMerge230135
def owner : Owner := ⟨.program ⟨257⟩, ⟨19142⟩⟩
def mergeEvent : Nat := 230135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events898.exact230129RawTerms
def group : MergeGroup := .relation 230131
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 230131) (rhsResult := 230129)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 230130 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19139⟩⟩]⟩) (none) 230129) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230135

namespace LeftMerge230140
def owner : Owner := ⟨.program ⟨257⟩, ⟨20210⟩⟩
def mergeEvent : Nat := 230140
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19703⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230136RawTerms
def rightRaw : List Term := Proof.Events898.exact229950RawTerms
def group : MergeGroup := .operator 230136 229950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230136) (leftOrdinal := 2)
    (rightResult := 229950) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19703⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19703⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨12666⟩⟩, ⟨.program ⟨257⟩, ⟨18250⟩⟩], [⟨.program ⟨257⟩, ⟨19703⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230140

namespace LeftMerge230141
def owner : Owner := ⟨.program ⟨257⟩, ⟨20210⟩⟩
def mergeEvent : Nat := 230141
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230136RawTerms
def rightRaw : List Term := Proof.Events898.exact229950RawTerms
def group : MergeGroup := .operator 230136 229950
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230136) (leftOrdinal := 1)
    (rightResult := 229950) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20208⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230141

namespace LeftMerge230149
def owner : Owner := ⟨.program ⟨257⟩, ⟨20623⟩⟩
def mergeEvent : Nat := 230149
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230143RawTerms
def rightRaw : List Term := Proof.Events897.exact229866RawTerms
def group : MergeGroup := .operator 230143 229866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230143) (leftOrdinal := 0)
    (rightResult := 229866) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20621⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge230149

namespace LeftMerge230150
def owner : Owner := ⟨.program ⟨257⟩, ⟨20623⟩⟩
def mergeEvent : Nat := 230150
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩] } }
def leftRaw : List Term := Proof.Events898.exact230143RawTerms
def rightRaw : List Term := Proof.Events897.exact229866RawTerms
def group : MergeGroup := .operator 230143 229866
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 230143) (leftOrdinal := 1)
    (rightResult := 229866) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20621⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨18580⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20621⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge230150

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
