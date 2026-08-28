import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge189616
def owner : Owner := ⟨.program ⟨257⟩, ⟨38235⟩⟩
def mergeEvent : Nat := 189616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38607⟩⟩] } }
def rhsRaw : List Term := Proof.Events740.exact189611RawTerms
def group : MergeGroup := .relation 189613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 189613) (rhsResult := 189611)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38232⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 189612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38232⟩⟩]⟩) (none) 189611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38607⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37452⟩⟩], [⟨.program ⟨257⟩, ⟨38607⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge189616

namespace LeftMerge189617
def owner : Owner := ⟨.program ⟨257⟩, ⟨38235⟩⟩
def mergeEvent : Nat := 189617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events740.exact189611RawTerms
def group : MergeGroup := .relation 189613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 189613) (rhsResult := 189611)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38232⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 189612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38232⟩⟩]⟩) (none) 189611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge189617

namespace LeftMerge189622
def owner : Owner := ⟨.program ⟨257⟩, ⟨39381⟩⟩
def mergeEvent : Nat := 189622
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39378⟩⟩] } }
def leftRaw : List Term := Proof.Events740.exact189618RawTerms
def rightRaw : List Term := Proof.Events740.exact189440RawTerms
def group : MergeGroup := .operator 189618 189440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 189618) (leftOrdinal := 0)
    (rightResult := 189440) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39378⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39378⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39378⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge189622

namespace LeftMerge189623
def owner : Owner := ⟨.program ⟨257⟩, ⟨39381⟩⟩
def mergeEvent : Nat := 189623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38607⟩⟩] } }
def leftRaw : List Term := Proof.Events740.exact189618RawTerms
def rightRaw : List Term := Proof.Events740.exact189440RawTerms
def group : MergeGroup := .operator 189618 189440
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 189618) (leftOrdinal := 2)
    (rightResult := 189440) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38607⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38607⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37452⟩⟩], [⟨.program ⟨257⟩, ⟨38607⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge189623

namespace LeftMerge189631
def owner : Owner := ⟨.program ⟨257⟩, ⟨39382⟩⟩
def mergeEvent : Nat := 189631
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩] } }
def leftRaw : List Term := Proof.Events740.exact189625RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 189625 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 189625) (leftOrdinal := 0)
    (rightResult := 15622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7161⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge189631

namespace LeftMerge189632
def owner : Owner := ⟨.program ⟨257⟩, ⟨39382⟩⟩
def mergeEvent : Nat := 189632
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩] } }
def leftRaw : List Term := Proof.Events740.exact189625RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 189625 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 189625) (leftOrdinal := 1)
    (rightResult := 15622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7161⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge189632

namespace LeftMerge189634
def owner : Owner := ⟨.program ⟨257⟩, ⟨39382⟩⟩
def mergeEvent : Nat := 189634
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15615RawTerms
def group : MergeGroup := .relation 189633
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 189633) (rhsResult := 15615)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7161⟩⟩) ⟨7046⟩ 15615) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge189634

namespace LeftMerge189648
def owner : Owner := ⟨.program ⟨257⟩, ⟨36700⟩⟩
def mergeEvent : Nat := 189648
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩] } }
def leftRaw : List Term := Proof.Events706.exact180966RawTerms
def rightRaw : List Term := Proof.Events740.exact189642RawTerms
def group : MergeGroup := .operator 180966 189642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180966) (leftOrdinal := 0)
    (rightResult := 189642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36698⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge189648

namespace LeftMerge189649
def owner : Owner := ⟨.program ⟨257⟩, ⟨36700⟩⟩
def mergeEvent : Nat := 189649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩] } }
def leftRaw : List Term := Proof.Events706.exact180966RawTerms
def rightRaw : List Term := Proof.Events740.exact189642RawTerms
def group : MergeGroup := .operator 180966 189642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180966) (leftOrdinal := 1)
    (rightResult := 189642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36698⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge189649

namespace LeftMerge189651
def owner : Owner := ⟨.program ⟨257⟩, ⟨36700⟩⟩
def mergeEvent : Nat := 189651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35927⟩⟩] } }
def rhsRaw : List Term := Proof.Events740.exact189639RawTerms
def group : MergeGroup := .relation 189650
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 189650) (rhsResult := 189639)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36698⟩⟩) ⟨35927⟩ 189639) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35927⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨35927⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge189651

namespace LeftMerge189665
def owner : Owner := ⟨.program ⟨257⟩, ⟨35555⟩⟩
def mergeEvent : Nat := 189665
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35552⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events740.exact189659RawTerms
def group : MergeGroup := .operator 178370 189659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 189659) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35552⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35552⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge189665

namespace LeftMerge189786
def owner : Owner := ⟨.program ⟨257⟩, ⟨36120⟩⟩
def mergeEvent : Nat := 189786
def frameStart : Nat := 189720
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events741.exact189782RawTerms
def rightRaw : List Term := Proof.Events741.exact189780RawTerms
def group : MergeGroup := .operator 189782 189780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 189782) (leftOrdinal := 0)
    (rightResult := 189780) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge189786

namespace LeftMerge189798
def owner : Owner := ⟨.program ⟨257⟩, ⟨36699⟩⟩
def mergeEvent : Nat := 189798
def frameStart : Nat := 189720
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩] } }
def leftRaw : List Term := Proof.Events741.exact189794RawTerms
def rightRaw : List Term := Proof.Events741.exact189771RawTerms
def group : MergeGroup := .operator 189794 189771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 189794) (leftOrdinal := 0)
    (rightResult := 189771) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36698⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge189798

namespace LeftMerge189799
def owner : Owner := ⟨.program ⟨257⟩, ⟨36699⟩⟩
def mergeEvent : Nat := 189799
def frameStart : Nat := 189720
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩] } }
def leftRaw : List Term := Proof.Events741.exact189794RawTerms
def rightRaw : List Term := Proof.Events741.exact189771RawTerms
def group : MergeGroup := .operator 189794 189771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 189794) (leftOrdinal := 1)
    (rightResult := 189771) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36698⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge189799

namespace LeftMerge189801
def owner : Owner := ⟨.program ⟨257⟩, ⟨36699⟩⟩
def mergeEvent : Nat := 189801
def frameStart : Nat := 189720
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35927⟩⟩] } }
def rhsRaw : List Term := Proof.Events741.exact189768RawTerms
def group : MergeGroup := .relation 189800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 189800) (rhsResult := 189768)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36698⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36698⟩⟩) ⟨35927⟩ 189768) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35927⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨35927⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge189801

namespace LeftMerge189809
def owner : Owner := ⟨.program ⟨257⟩, ⟨35000⟩⟩
def mergeEvent : Nat := 189809
def frameStart : Nat := 189720
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events741.exact189782RawTerms
def rightRaw : List Term := Proof.Events741.exact189805RawTerms
def group : MergeGroup := .operator 189782 189805
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 189782) (leftOrdinal := 0)
    (rightResult := 189805) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34998⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge189809

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
