import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge167748
def owner : Owner := ⟨.program ⟨257⟩, ⟨69287⟩⟩
def mergeEvent : Nat := 167748
def frameStart : Nat := 167656
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167744RawTerms
def rightRaw : List Term := Proof.Events655.exact167701RawTerms
def group : MergeGroup := .operator 167744 167701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167744) (leftOrdinal := 0)
    (rightResult := 167701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69284⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167748

namespace LeftMerge167749
def owner : Owner := ⟨.program ⟨257⟩, ⟨69287⟩⟩
def mergeEvent : Nat := 167749
def frameStart : Nat := 167656
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167744RawTerms
def rightRaw : List Term := Proof.Events655.exact167701RawTerms
def group : MergeGroup := .operator 167744 167701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167744) (leftOrdinal := 1)
    (rightResult := 167701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167749

namespace LeftMerge167751
def owner : Owner := ⟨.program ⟨257⟩, ⟨69287⟩⟩
def mergeEvent : Nat := 167751
def frameStart : Nat := 167656
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68554⟩⟩] } }
def rhsRaw : List Term := Proof.Events655.exact167698RawTerms
def group : MergeGroup := .relation 167750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167750) (rhsResult := 167698)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69284⟩⟩) ⟨68554⟩ 167698) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68554⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], [⟨.program ⟨257⟩, ⟨68554⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167751

namespace LeftMerge167759
def owner : Owner := ⟨.program ⟨257⟩, ⟨65822⟩⟩
def mergeEvent : Nat := 167759
def frameStart : Nat := 167656
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167712RawTerms
def rightRaw : List Term := Proof.Events655.exact167755RawTerms
def group : MergeGroup := .operator 167712 167755
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167712) (leftOrdinal := 0)
    (rightResult := 167755) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167759

namespace LeftMerge167776
def owner : Owner := ⟨.program ⟨257⟩, ⟨67813⟩⟩
def mergeEvent : Nat := 167776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events655.exact167773RawTerms
def group : MergeGroup := .relation 167775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167775) (rhsResult := 167773)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67810⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 167774 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67810⟩⟩]⟩) (none) 167773) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167776

namespace LeftMerge167777
def owner : Owner := ⟨.program ⟨257⟩, ⟨67813⟩⟩
def mergeEvent : Nat := 167777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩] } }
def rhsRaw : List Term := Proof.Events655.exact167773RawTerms
def group : MergeGroup := .relation 167775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167775) (rhsResult := 167773)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67810⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 167774 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67810⟩⟩]⟩) (none) 167773) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167777

namespace LeftMerge167778
def owner : Owner := ⟨.program ⟨257⟩, ⟨67813⟩⟩
def mergeEvent : Nat := 167778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68554⟩⟩] } }
def rhsRaw : List Term := Proof.Events655.exact167773RawTerms
def group : MergeGroup := .relation 167775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167775) (rhsResult := 167773)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67810⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 167774 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67810⟩⟩]⟩) (none) 167773) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68554⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], [⟨.program ⟨257⟩, ⟨68554⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167778

namespace LeftMerge167779
def owner : Owner := ⟨.program ⟨257⟩, ⟨67813⟩⟩
def mergeEvent : Nat := 167779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events655.exact167773RawTerms
def group : MergeGroup := .relation 167775
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167775) (rhsResult := 167773)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67810⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 167774 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67810⟩⟩]⟩) (none) 167773) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167779

namespace LeftMerge167784
def owner : Owner := ⟨.program ⟨257⟩, ⟨69286⟩⟩
def mergeEvent : Nat := 167784
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68554⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167780RawTerms
def rightRaw : List Term := Proof.Events654.exact167594RawTerms
def group : MergeGroup := .operator 167780 167594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167780) (leftOrdinal := 2)
    (rightResult := 167594) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68554⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68554⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25778⟩⟩, ⟨.program ⟨257⟩, ⟨65553⟩⟩], [⟨.program ⟨257⟩, ⟨68554⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167784

namespace LeftMerge167785
def owner : Owner := ⟨.program ⟨257⟩, ⟨69286⟩⟩
def mergeEvent : Nat := 167785
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167780RawTerms
def rightRaw : List Term := Proof.Events654.exact167594RawTerms
def group : MergeGroup := .operator 167780 167594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167780) (leftOrdinal := 1)
    (rightResult := 167594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167785

namespace LeftMerge167793
def owner : Owner := ⟨.program ⟨257⟩, ⟨70495⟩⟩
def mergeEvent : Nat := 167793
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70493⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167787RawTerms
def rightRaw : List Term := Proof.Events654.exact167510RawTerms
def group : MergeGroup := .operator 167787 167510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167787) (leftOrdinal := 0)
    (rightResult := 167510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70493⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70493⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167793

namespace LeftMerge167794
def owner : Owner := ⟨.program ⟨257⟩, ⟨70495⟩⟩
def mergeEvent : Nat := 167794
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70493⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167787RawTerms
def rightRaw : List Term := Proof.Events654.exact167510RawTerms
def group : MergeGroup := .operator 167787 167510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167787) (leftOrdinal := 1)
    (rightResult := 167510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70493⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70493⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167794

namespace LeftMerge167796
def owner : Owner := ⟨.program ⟨257⟩, ⟨70495⟩⟩
def mergeEvent : Nat := 167796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68718⟩⟩] } }
def rhsRaw : List Term := Proof.Events654.exact167507RawTerms
def group : MergeGroup := .relation 167795
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 167795) (rhsResult := 167507)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70493⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70493⟩⟩) ⟨68718⟩ 167507) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68718⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨68718⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge167796

namespace LeftMerge167810
def owner : Owner := ⟨.program ⟨257⟩, ⟨68160⟩⟩
def mergeEvent : Nat := 167810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68157⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events655.exact167804RawTerms
def group : MergeGroup := .operator 163745 167804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 167804) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68157⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68157⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167810

namespace LeftMerge167931
def owner : Owner := ⟨.program ⟨257⟩, ⟨69025⟩⟩
def mergeEvent : Nat := 167931
def frameStart : Nat := 167865
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events655.exact167927RawTerms
def rightRaw : List Term := Proof.Events655.exact167925RawTerms
def group : MergeGroup := .operator 167927 167925
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167927) (leftOrdinal := 0)
    (rightResult := 167925) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167931

namespace LeftMerge167943
def owner : Owner := ⟨.program ⟨257⟩, ⟨70494⟩⟩
def mergeEvent : Nat := 167943
def frameStart : Nat := 167865
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70493⟩⟩] } }
def leftRaw : List Term := Proof.Events656.exact167939RawTerms
def rightRaw : List Term := Proof.Events655.exact167916RawTerms
def group : MergeGroup := .operator 167939 167916
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 167939) (leftOrdinal := 0)
    (rightResult := 167916) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70493⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70493⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge167943

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
