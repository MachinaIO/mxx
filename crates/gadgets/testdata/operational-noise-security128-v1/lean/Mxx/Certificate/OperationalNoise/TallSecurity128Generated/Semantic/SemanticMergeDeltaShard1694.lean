import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge274810
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 11)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274810

namespace LeftMerge274811
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 22)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274811

namespace LeftMerge274813
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1039.exact266000RawTerms
def group : MergeGroup := .relation 274812
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274812) (rhsResult := 266000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨29192⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274813

namespace LeftMerge274814
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 10)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274814

namespace LeftMerge274815
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274815
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 21)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274815

namespace LeftMerge274817
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274817
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1039.exact266000RawTerms
def group : MergeGroup := .relation 274816
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274816) (rhsResult := 266000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26512⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274817

namespace LeftMerge274818
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274818
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 9)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274818

namespace LeftMerge274819
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274819
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 35)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274819

namespace LeftMerge274821
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274821
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1039.exact266000RawTerms
def group : MergeGroup := .relation 274820
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274820) (rhsResult := 266000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨66019⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274821

namespace LeftMerge274822
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274822
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 8)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274822

namespace LeftMerge274823
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274823
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 34)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274823

namespace LeftMerge274825
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274825
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1039.exact266000RawTerms
def group : MergeGroup := .relation 274824
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274824) (rhsResult := 266000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨62924⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274825

namespace LeftMerge274826
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274826
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 7)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274826

namespace LeftMerge274827
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 33)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274827

namespace LeftMerge274829
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }
def rhsRaw : List Term := Proof.Events1039.exact266000RawTerms
def group : MergeGroup := .relation 274828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274828) (rhsResult := 266000)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70979⟩⟩) ⟨68780⟩ 266000) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68780⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨59944⟩⟩], [⟨.program ⟨257⟩, ⟨68780⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274829

namespace LeftMerge274830
def owner : Owner := ⟨.program ⟨257⟩, ⟨70981⟩⟩
def mergeEvent : Nat := 274830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩] } }
def leftRaw : List Term := Proof.Events1073.exact274780RawTerms
def rightRaw : List Term := Proof.Events1039.exact266003RawTerms
def group : MergeGroup := .operator 274780 266003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274780) (leftOrdinal := 6)
    (rightResult := 266003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274830

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
