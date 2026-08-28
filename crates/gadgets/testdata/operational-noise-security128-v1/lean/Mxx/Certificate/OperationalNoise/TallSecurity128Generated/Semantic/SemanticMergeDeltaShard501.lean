import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge84697
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84697
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 8)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84697

namespace LeftMerge84698
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84698
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 34)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84698

namespace LeftMerge84700
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84700
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events296.exact75875RawTerms
def group : MergeGroup := .relation 84699
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84699) (rhsResult := 75875)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨63195⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84700

namespace LeftMerge84701
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84701
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 7)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84701

namespace LeftMerge84702
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84702
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 33)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84702

namespace LeftMerge84704
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events296.exact75875RawTerms
def group : MergeGroup := .relation 84703
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84703) (rhsResult := 75875)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨60215⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84704

namespace LeftMerge84705
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84705
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 6)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84705

namespace LeftMerge84706
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 32)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84706

namespace LeftMerge84708
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84708
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events296.exact75875RawTerms
def group : MergeGroup := .relation 84707
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84707) (rhsResult := 75875)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨57235⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84708

namespace LeftMerge84709
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84709
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 5)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84709

namespace LeftMerge84710
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84710
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 31)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84710

namespace LeftMerge84712
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84712
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events296.exact75875RawTerms
def group : MergeGroup := .relation 84711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84711) (rhsResult := 75875)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨54255⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84712

namespace LeftMerge84713
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84713
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 4)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84713

namespace LeftMerge84714
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84714
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 30)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84714

namespace LeftMerge84716
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84716
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }
def rhsRaw : List Term := Proof.Events296.exact75875RawTerms
def group : MergeGroup := .relation 84715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 84715) (rhsResult := 75875)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71437⟩⟩) ⟨68866⟩ 75875) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68866⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51275⟩⟩], [⟨.program ⟨257⟩, ⟨68866⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge84716

namespace LeftMerge84717
def owner : Owner := ⟨.program ⟨257⟩, ⟨71439⟩⟩
def mergeEvent : Nat := 84717
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩] } }
def leftRaw : List Term := Proof.Events330.exact84655RawTerms
def rightRaw : List Term := Proof.Events296.exact75878RawTerms
def group : MergeGroup := .operator 84655 75878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 84655) (leftOrdinal := 3)
    (rightResult := 75878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71437⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71437⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge84717

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
