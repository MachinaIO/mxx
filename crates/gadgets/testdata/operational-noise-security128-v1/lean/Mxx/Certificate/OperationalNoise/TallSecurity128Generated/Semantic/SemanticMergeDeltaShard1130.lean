import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge184784
def owner : Owner := ⟨.program ⟨257⟩, ⟨52555⟩⟩
def mergeEvent : Nat := 184784
def frameStart : Nat := 184691
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩] } }
def leftRaw : List Term := Proof.Events721.exact184779RawTerms
def rightRaw : List Term := Proof.Events721.exact184736RawTerms
def group : MergeGroup := .operator 184779 184736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184779) (leftOrdinal := 1)
    (rightResult := 184736) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52552⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184784

namespace LeftMerge184786
def owner : Owner := ⟨.program ⟨257⟩, ⟨52555⟩⟩
def mergeEvent : Nat := 184786
def frameStart : Nat := 184691
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52027⟩⟩] } }
def rhsRaw : List Term := Proof.Events721.exact184733RawTerms
def group : MergeGroup := .relation 184785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184785) (rhsResult := 184733)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52552⟩⟩) ⟨52027⟩ 184733) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52027⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], [⟨.program ⟨257⟩, ⟨52027⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184786

namespace LeftMerge184794
def owner : Owner := ⟨.program ⟨257⟩, ⟨50914⟩⟩
def mergeEvent : Nat := 184794
def frameStart : Nat := 184691
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events721.exact184747RawTerms
def rightRaw : List Term := Proof.Events721.exact184790RawTerms
def group : MergeGroup := .operator 184747 184790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184747) (leftOrdinal := 0)
    (rightResult := 184790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184794

namespace LeftMerge184811
def owner : Owner := ⟨.program ⟨257⟩, ⟨51482⟩⟩
def mergeEvent : Nat := 184811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events721.exact184808RawTerms
def group : MergeGroup := .relation 184810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184810) (rhsResult := 184808)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51479⟩⟩]⟩) (none) 184808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184811

namespace LeftMerge184812
def owner : Owner := ⟨.program ⟨257⟩, ⟨51482⟩⟩
def mergeEvent : Nat := 184812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩] } }
def rhsRaw : List Term := Proof.Events721.exact184808RawTerms
def group : MergeGroup := .relation 184810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184810) (rhsResult := 184808)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51479⟩⟩]⟩) (none) 184808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184812

namespace LeftMerge184813
def owner : Owner := ⟨.program ⟨257⟩, ⟨51482⟩⟩
def mergeEvent : Nat := 184813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52027⟩⟩] } }
def rhsRaw : List Term := Proof.Events721.exact184808RawTerms
def group : MergeGroup := .relation 184810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184810) (rhsResult := 184808)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51479⟩⟩]⟩) (none) 184808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52027⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], [⟨.program ⟨257⟩, ⟨52027⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184813

namespace LeftMerge184814
def owner : Owner := ⟨.program ⟨257⟩, ⟨51482⟩⟩
def mergeEvent : Nat := 184814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events721.exact184808RawTerms
def group : MergeGroup := .relation 184810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184810) (rhsResult := 184808)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 184809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51479⟩⟩]⟩) (none) 184808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨50912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184814

namespace LeftMerge184819
def owner : Owner := ⟨.program ⟨257⟩, ⟨52554⟩⟩
def mergeEvent : Nat := 184819
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52027⟩⟩] } }
def leftRaw : List Term := Proof.Events721.exact184815RawTerms
def rightRaw : List Term := Proof.Events721.exact184629RawTerms
def group : MergeGroup := .operator 184815 184629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184815) (leftOrdinal := 2)
    (rightResult := 184629) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52027⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52027⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨24566⟩⟩, ⟨.program ⟨257⟩, ⟨50626⟩⟩], [⟨.program ⟨257⟩, ⟨52027⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184819

namespace LeftMerge184820
def owner : Owner := ⟨.program ⟨257⟩, ⟨52554⟩⟩
def mergeEvent : Nat := 184820
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩] } }
def leftRaw : List Term := Proof.Events721.exact184815RawTerms
def rightRaw : List Term := Proof.Events721.exact184629RawTerms
def group : MergeGroup := .operator 184815 184629
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184815) (leftOrdinal := 1)
    (rightResult := 184629) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52552⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184820

namespace LeftMerge184828
def owner : Owner := ⟨.program ⟨257⟩, ⟨53047⟩⟩
def mergeEvent : Nat := 184828
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53045⟩⟩] } }
def leftRaw : List Term := Proof.Events721.exact184822RawTerms
def rightRaw : List Term := Proof.Events720.exact184545RawTerms
def group : MergeGroup := .operator 184822 184545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184822) (leftOrdinal := 0)
    (rightResult := 184545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53045⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53045⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184828

namespace LeftMerge184829
def owner : Owner := ⟨.program ⟨257⟩, ⟨53047⟩⟩
def mergeEvent : Nat := 184829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53045⟩⟩] } }
def leftRaw : List Term := Proof.Events721.exact184822RawTerms
def rightRaw : List Term := Proof.Events720.exact184545RawTerms
def group : MergeGroup := .operator 184822 184545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184822) (leftOrdinal := 1)
    (rightResult := 184545) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53045⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨50912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53045⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184829

namespace LeftMerge184831
def owner : Owner := ⟨.program ⟨257⟩, ⟨53047⟩⟩
def mergeEvent : Nat := 184831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52188⟩⟩] } }
def rhsRaw : List Term := Proof.Events720.exact184542RawTerms
def group : MergeGroup := .relation 184830
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 184830) (rhsResult := 184542)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨50912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53045⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53045⟩⟩) ⟨52188⟩ 184542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨50912⟩⟩], [⟨.program ⟨257⟩, ⟨52188⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184831

namespace LeftMerge184845
def owner : Owner := ⟨.program ⟨257⟩, ⟨51819⟩⟩
def mergeEvent : Nat := 184845
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51816⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events722.exact184839RawTerms
def group : MergeGroup := .operator 178370 184839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 184839) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51816⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51816⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184845

namespace LeftMerge184966
def owner : Owner := ⟨.program ⟨257⟩, ⟨52380⟩⟩
def mergeEvent : Nat := 184966
def frameStart : Nat := 184900
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events722.exact184962RawTerms
def rightRaw : List Term := Proof.Events722.exact184960RawTerms
def group : MergeGroup := .operator 184962 184960
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184962) (leftOrdinal := 0)
    (rightResult := 184960) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184966

namespace LeftMerge184978
def owner : Owner := ⟨.program ⟨257⟩, ⟨53046⟩⟩
def mergeEvent : Nat := 184978
def frameStart : Nat := 184900
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53045⟩⟩] } }
def leftRaw : List Term := Proof.Events722.exact184974RawTerms
def rightRaw : List Term := Proof.Events722.exact184951RawTerms
def group : MergeGroup := .operator 184974 184951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184974) (leftOrdinal := 0)
    (rightResult := 184951) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53045⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53045⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge184978

namespace LeftMerge184979
def owner : Owner := ⟨.program ⟨257⟩, ⟨53046⟩⟩
def mergeEvent : Nat := 184979
def frameStart : Nat := 184900
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53045⟩⟩] } }
def leftRaw : List Term := Proof.Events722.exact184974RawTerms
def rightRaw : List Term := Proof.Events722.exact184951RawTerms
def group : MergeGroup := .operator 184974 184951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184974) (leftOrdinal := 1)
    (rightResult := 184951) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50912⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53045⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50912⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53045⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge184979

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
