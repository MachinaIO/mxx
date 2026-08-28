import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge90028
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90028
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18205⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events350.exact89830RawTerms
def group : MergeGroup := .relation 90027
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90027) (rhsResult := 89830)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 89830) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18205⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90028

namespace LeftMerge90029
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90029
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 32)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90029

namespace LeftMerge90031
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90031
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17904⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events350.exact89830RawTerms
def group : MergeGroup := .relation 90030
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90030) (rhsResult := 89830)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 89830) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17904⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90031

namespace LeftMerge90032
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90032
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 30)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90032

namespace LeftMerge90034
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90034
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17120⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events350.exact89830RawTerms
def group : MergeGroup := .relation 90033
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90033) (rhsResult := 89830)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 89830) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17120⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90034

namespace LeftMerge90035
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90035
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 26)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90035

namespace LeftMerge90037
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90037
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16308⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events350.exact89830RawTerms
def group : MergeGroup := .relation 90036
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90036) (rhsResult := 89830)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 89830) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16308⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90037

namespace LeftMerge90038
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90038
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 35)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90038

namespace LeftMerge90040
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90040
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18340⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events350.exact89830RawTerms
def group : MergeGroup := .relation 90039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90039) (rhsResult := 89830)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 89830) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18340⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90040

namespace LeftMerge90041
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90041
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 25)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90041

namespace LeftMerge90043
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90043
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16105⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events350.exact89830RawTerms
def group : MergeGroup := .relation 90042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90042) (rhsResult := 89830)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 89830) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16105⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90043

namespace LeftMerge90044
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90044
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 24)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90044

namespace LeftMerge90046
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90046
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15986⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events350.exact89830RawTerms
def group : MergeGroup := .relation 90045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90045) (rhsResult := 89830)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 89830) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15986⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90046

namespace LeftMerge90047
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90047
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 23)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90047

namespace LeftMerge90049
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90049
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15867⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }
def rhsRaw : List Term := Proof.Events350.exact89830RawTerms
def group : MergeGroup := .relation 90048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90048) (rhsResult := 89830)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18681⟩⟩) ⟨18618⟩ 89830) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18618⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15867⟩⟩], [⟨.program ⟨214⟩, ⟨18618⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90049

namespace LeftMerge90050
def owner : Owner := ⟨.program ⟨214⟩, ⟨18682⟩⟩
def mergeEvent : Nat := 90050
def frameStart : Nat := 89317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩] } }
def leftRaw : List Term := Proof.Events351.exact89992RawTerms
def rightRaw : List Term := Proof.Events350.exact89833RawTerms
def group : MergeGroup := .operator 89992 89833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 89992) (leftOrdinal := 22)
    (rightResult := 89833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18681⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15748⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90050

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
