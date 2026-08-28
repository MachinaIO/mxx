import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge17651
def owner : Owner := ⟨.program ⟨257⟩, ⟨44953⟩⟩
def mergeEvent : Nat := 17651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events068.exact17647RawTerms
def rightRaw : List Term := Proof.Events068.exact17604RawTerms
def group : MergeGroup := .operator 17647 17604
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17647) (leftOrdinal := 1)
    (rightResult := 17604) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17651

namespace LeftMerge17659
def owner : Owner := ⟨.program ⟨257⟩, ⟨46884⟩⟩
def mergeEvent : Nat := 17659
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩] } }
def leftRaw : List Term := Proof.Events068.exact17653RawTerms
def rightRaw : List Term := Proof.Events068.exact17570RawTerms
def group : MergeGroup := .operator 17653 17570
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17653) (leftOrdinal := 1)
    (rightResult := 17570) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46883⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17659

namespace LeftMerge17661
def owner : Owner := ⟨.program ⟨257⟩, ⟨46884⟩⟩
def mergeEvent : Nat := 17661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46417⟩⟩] } }
def rhsRaw : List Term := Proof.Events068.exact17567RawTerms
def group : MergeGroup := .relation 17660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17660) (rhsResult := 17567)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46883⟩⟩) ⟨46417⟩ 17567) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46417⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨46417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17661

namespace LeftMerge17662
def owner : Owner := ⟨.program ⟨257⟩, ⟨46884⟩⟩
def mergeEvent : Nat := 17662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩] } }
def leftRaw : List Term := Proof.Events068.exact17653RawTerms
def rightRaw : List Term := Proof.Events068.exact17570RawTerms
def group : MergeGroup := .operator 17653 17570
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17653) (leftOrdinal := 0)
    (rightResult := 17570) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46883⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17662

namespace LeftMerge17676
def owner : Owner := ⟨.program ⟨257⟩, ⟨45825⟩⟩
def mergeEvent : Nat := 17676
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events069.exact17670RawTerms
def group : MergeGroup := .operator 17169 17670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 17670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45822⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17676

namespace LeftMerge17755
def owner : Owner := ⟨.program ⟨257⟩, ⟨44947⟩⟩
def mergeEvent : Nat := 17755
def frameStart : Nat := 17725
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events069.exact17751RawTerms
def rightRaw : List Term := Proof.Events069.exact17748RawTerms
def group : MergeGroup := .operator 17751 17748
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17751) (leftOrdinal := 0)
    (rightResult := 17748) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14651⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17755

namespace LeftMerge17785
def owner : Owner := ⟨.program ⟨257⟩, ⟨46712⟩⟩
def mergeEvent : Nat := 17785
def frameStart : Nat := 17725
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17781RawTerms
def rightRaw : List Term := Proof.Events069.exact17779RawTerms
def group : MergeGroup := .operator 17781 17779
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17781) (leftOrdinal := 0)
    (rightResult := 17779) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17785

namespace LeftMerge17808
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 17808
def frameStart : Nat := 17725
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17804RawTerms
def rightRaw : List Term := Proof.Events069.exact17801RawTerms
def group : MergeGroup := .operator 17804 17801
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17804) (leftOrdinal := 0)
    (rightResult := 17801) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17808

namespace LeftMerge17817
def owner : Owner := ⟨.program ⟨257⟩, ⟨46886⟩⟩
def mergeEvent : Nat := 17817
def frameStart : Nat := 17725
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17813RawTerms
def rightRaw : List Term := Proof.Events069.exact17770RawTerms
def group : MergeGroup := .operator 17813 17770
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17813) (leftOrdinal := 1)
    (rightResult := 17770) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46883⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17817

namespace LeftMerge17819
def owner : Owner := ⟨.program ⟨257⟩, ⟨46886⟩⟩
def mergeEvent : Nat := 17819
def frameStart : Nat := 17725
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46417⟩⟩] } }
def rhsRaw : List Term := Proof.Events069.exact17767RawTerms
def group : MergeGroup := .relation 17818
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17818) (rhsResult := 17767)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨46883⟩⟩) ⟨46417⟩ 17767) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46417⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨46417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17819

namespace LeftMerge17820
def owner : Owner := ⟨.program ⟨257⟩, ⟨46886⟩⟩
def mergeEvent : Nat := 17820
def frameStart : Nat := 17725
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17813RawTerms
def rightRaw : List Term := Proof.Events069.exact17770RawTerms
def group : MergeGroup := .operator 17813 17770
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17813) (leftOrdinal := 0)
    (rightResult := 17770) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46883⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17820

namespace LeftMerge17828
def owner : Owner := ⟨.program ⟨257⟩, ⟨45400⟩⟩
def mergeEvent : Nat := 17828
def frameStart : Nat := 17725
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events069.exact17781RawTerms
def rightRaw : List Term := Proof.Events069.exact17824RawTerms
def group : MergeGroup := .operator 17781 17824
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17781) (leftOrdinal := 0)
    (rightResult := 17824) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17828

namespace LeftMerge17845
def owner : Owner := ⟨.program ⟨257⟩, ⟨45825⟩⟩
def mergeEvent : Nat := 17845
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46417⟩⟩] } }
def rhsRaw : List Term := Proof.Events069.exact17842RawTerms
def group : MergeGroup := .relation 17844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17844) (rhsResult := 17842)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 17843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩]⟩) (none) 17842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46417⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14651⟩⟩, ⟨.program ⟨257⟩, ⟨44946⟩⟩], [⟨.program ⟨257⟩, ⟨46417⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17845

namespace LeftMerge17846
def owner : Owner := ⟨.program ⟨257⟩, ⟨45825⟩⟩
def mergeEvent : Nat := 17846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩] } }
def rhsRaw : List Term := Proof.Events069.exact17842RawTerms
def group : MergeGroup := .relation 17844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17844) (rhsResult := 17842)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 17843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩]⟩) (none) 17842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46883⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17846

namespace LeftMerge17847
def owner : Owner := ⟨.program ⟨257⟩, ⟨45825⟩⟩
def mergeEvent : Nat := 17847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events069.exact17842RawTerms
def group : MergeGroup := .relation 17844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17844) (rhsResult := 17842)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 17843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩]⟩) (none) 17842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45398⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨45398⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17847

namespace LeftMerge17848
def owner : Owner := ⟨.program ⟨257⟩, ⟨45825⟩⟩
def mergeEvent : Nat := 17848
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events069.exact17842RawTerms
def group : MergeGroup := .relation 17844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17844) (rhsResult := 17842)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 17843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45822⟩⟩]⟩) (none) 17842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17848

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
