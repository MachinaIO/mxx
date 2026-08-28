import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge164715
def owner : Owner := ⟨.program ⟨257⟩, ⟨43272⟩⟩
def mergeEvent : Nat := 164715
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events643.exact164709RawTerms
def group : MergeGroup := .operator 163745 164709
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 164709) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164715

namespace LeftMerge164794
def owner : Owner := ⟨.program ⟨257⟩, ⟨42571⟩⟩
def mergeEvent : Nat := 164794
def frameStart : Nat := 164764
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events643.exact164790RawTerms
def rightRaw : List Term := Proof.Events643.exact164787RawTerms
def group : MergeGroup := .operator 164790 164787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164790) (leftOrdinal := 0)
    (rightResult := 164787) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14541⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164794

namespace LeftMerge164824
def owner : Owner := ⟨.program ⟨257⟩, ⟨44084⟩⟩
def mergeEvent : Nat := 164824
def frameStart : Nat := 164764
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events643.exact164820RawTerms
def rightRaw : List Term := Proof.Events643.exact164818RawTerms
def group : MergeGroup := .operator 164820 164818
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164820) (leftOrdinal := 0)
    (rightResult := 164818) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164824

namespace LeftMerge164847
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def mergeEvent : Nat := 164847
def frameStart : Nat := 164764
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events643.exact164843RawTerms
def rightRaw : List Term := Proof.Events643.exact164840RawTerms
def group : MergeGroup := .operator 164843 164840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164843) (leftOrdinal := 0)
    (rightResult := 164840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164847

namespace LeftMerge164856
def owner : Owner := ⟨.program ⟨257⟩, ⟨44346⟩⟩
def mergeEvent : Nat := 164856
def frameStart : Nat := 164764
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩] } }
def leftRaw : List Term := Proof.Events643.exact164852RawTerms
def rightRaw : List Term := Proof.Events643.exact164809RawTerms
def group : MergeGroup := .operator 164852 164809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164852) (leftOrdinal := 0)
    (rightResult := 164809) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44343⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164856

namespace LeftMerge164857
def owner : Owner := ⟨.program ⟨257⟩, ⟨44346⟩⟩
def mergeEvent : Nat := 164857
def frameStart : Nat := 164764
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩] } }
def leftRaw : List Term := Proof.Events643.exact164852RawTerms
def rightRaw : List Term := Proof.Events643.exact164809RawTerms
def group : MergeGroup := .operator 164852 164809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164852) (leftOrdinal := 1)
    (rightResult := 164809) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44343⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164857

namespace LeftMerge164859
def owner : Owner := ⟨.program ⟨257⟩, ⟨44346⟩⟩
def mergeEvent : Nat := 164859
def frameStart : Nat := 164764
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43813⟩⟩] } }
def rhsRaw : List Term := Proof.Events643.exact164806RawTerms
def group : MergeGroup := .relation 164858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164858) (rhsResult := 164806)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44343⟩⟩) ⟨43813⟩ 164806) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43813⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], [⟨.program ⟨257⟩, ⟨43813⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164859

namespace LeftMerge164867
def owner : Owner := ⟨.program ⟨257⟩, ⟨42822⟩⟩
def mergeEvent : Nat := 164867
def frameStart : Nat := 164764
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events643.exact164820RawTerms
def rightRaw : List Term := Proof.Events643.exact164863RawTerms
def group : MergeGroup := .operator 164820 164863
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164820) (leftOrdinal := 0)
    (rightResult := 164863) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164867

namespace LeftMerge164884
def owner : Owner := ⟨.program ⟨257⟩, ⟨43272⟩⟩
def mergeEvent : Nat := 164884
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events644.exact164881RawTerms
def group : MergeGroup := .relation 164883
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164883) (rhsResult := 164881)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164882 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩]⟩) (none) 164881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164884

namespace LeftMerge164885
def owner : Owner := ⟨.program ⟨257⟩, ⟨43272⟩⟩
def mergeEvent : Nat := 164885
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩] } }
def rhsRaw : List Term := Proof.Events644.exact164881RawTerms
def group : MergeGroup := .relation 164883
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164883) (rhsResult := 164881)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164882 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩]⟩) (none) 164881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164885

namespace LeftMerge164886
def owner : Owner := ⟨.program ⟨257⟩, ⟨43272⟩⟩
def mergeEvent : Nat := 164886
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43813⟩⟩] } }
def rhsRaw : List Term := Proof.Events644.exact164881RawTerms
def group : MergeGroup := .relation 164883
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164883) (rhsResult := 164881)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164882 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩]⟩) (none) 164881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43813⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], [⟨.program ⟨257⟩, ⟨43813⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164886

namespace LeftMerge164887
def owner : Owner := ⟨.program ⟨257⟩, ⟨43272⟩⟩
def mergeEvent : Nat := 164887
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events644.exact164881RawTerms
def group : MergeGroup := .relation 164883
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164883) (rhsResult := 164881)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164882 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43269⟩⟩]⟩) (none) 164881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164887

namespace LeftMerge164892
def owner : Owner := ⟨.program ⟨257⟩, ⟨44345⟩⟩
def mergeEvent : Nat := 164892
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43813⟩⟩] } }
def leftRaw : List Term := Proof.Events644.exact164888RawTerms
def rightRaw : List Term := Proof.Events643.exact164702RawTerms
def group : MergeGroup := .operator 164888 164702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164888) (leftOrdinal := 2)
    (rightResult := 164702) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43813⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43813⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14541⟩⟩, ⟨.program ⟨257⟩, ⟨42570⟩⟩], [⟨.program ⟨257⟩, ⟨43813⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164892

namespace LeftMerge164893
def owner : Owner := ⟨.program ⟨257⟩, ⟨44345⟩⟩
def mergeEvent : Nat := 164893
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩] } }
def leftRaw : List Term := Proof.Events644.exact164888RawTerms
def rightRaw : List Term := Proof.Events643.exact164702RawTerms
def group : MergeGroup := .operator 164888 164702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164888) (leftOrdinal := 1)
    (rightResult := 164702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44343⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164893

namespace LeftMerge164901
def owner : Owner := ⟨.program ⟨257⟩, ⟨44771⟩⟩
def mergeEvent : Nat := 164901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44769⟩⟩] } }
def leftRaw : List Term := Proof.Events644.exact164895RawTerms
def rightRaw : List Term := Proof.Events643.exact164618RawTerms
def group : MergeGroup := .operator 164895 164618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164895) (leftOrdinal := 0)
    (rightResult := 164618) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44769⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44769⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164901

namespace LeftMerge164902
def owner : Owner := ⟨.program ⟨257⟩, ⟨44771⟩⟩
def mergeEvent : Nat := 164902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44769⟩⟩] } }
def leftRaw : List Term := Proof.Events644.exact164895RawTerms
def rightRaw : List Term := Proof.Events643.exact164618RawTerms
def group : MergeGroup := .operator 164895 164618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164895) (leftOrdinal := 1)
    (rightResult := 164618) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44769⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨42820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44769⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164902

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
