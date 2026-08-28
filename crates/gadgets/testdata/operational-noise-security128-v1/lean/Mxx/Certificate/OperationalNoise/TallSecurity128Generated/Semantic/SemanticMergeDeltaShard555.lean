import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge92537
def owner : Owner := ⟨.program ⟨257⟩, ⟨38995⟩⟩
def mergeEvent : Nat := 92537
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩] } }
def leftRaw : List Term := Proof.Events361.exact92531RawTerms
def rightRaw : List Term := Proof.Events361.exact92467RawTerms
def group : MergeGroup := .operator 92531 92467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92531) (leftOrdinal := 1)
    (rightResult := 92467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92537

namespace LeftMerge92539
def owner : Owner := ⟨.program ⟨257⟩, ⟨38995⟩⟩
def mergeEvent : Nat := 92539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38459⟩⟩] } }
def rhsRaw : List Term := Proof.Events361.exact92464RawTerms
def group : MergeGroup := .relation 92538
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92538) (rhsResult := 92464)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38994⟩⟩) ⟨38459⟩ 92464) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38459⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], [⟨.program ⟨257⟩, ⟨38459⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92539

namespace LeftMerge92540
def owner : Owner := ⟨.program ⟨257⟩, ⟨38995⟩⟩
def mergeEvent : Nat := 92540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩] } }
def leftRaw : List Term := Proof.Events361.exact92531RawTerms
def rightRaw : List Term := Proof.Events361.exact92467RawTerms
def group : MergeGroup := .operator 92531 92467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92531) (leftOrdinal := 0)
    (rightResult := 92467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92540

namespace LeftMerge92554
def owner : Owner := ⟨.program ⟨257⟩, ⟨37922⟩⟩
def mergeEvent : Nat := 92554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events361.exact92548RawTerms
def group : MergeGroup := .operator 90620 92548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 92548) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92554

namespace LeftMerge92633
def owner : Owner := ⟨.program ⟨257⟩, ⟨37235⟩⟩
def mergeEvent : Nat := 92633
def frameStart : Nat := 92603
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events361.exact92629RawTerms
def rightRaw : List Term := Proof.Events361.exact92626RawTerms
def group : MergeGroup := .operator 92629 92626
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92629) (leftOrdinal := 0)
    (rightResult := 92626) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13956⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92633

namespace LeftMerge92663
def owner : Owner := ⟨.program ⟨257⟩, ⟨38728⟩⟩
def mergeEvent : Nat := 92663
def frameStart : Nat := 92603
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events361.exact92659RawTerms
def rightRaw : List Term := Proof.Events361.exact92657RawTerms
def group : MergeGroup := .operator 92659 92657
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92659) (leftOrdinal := 0)
    (rightResult := 92657) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92663

namespace LeftMerge92686
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 92686
def frameStart : Nat := 92603
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events362.exact92682RawTerms
def rightRaw : List Term := Proof.Events362.exact92679RawTerms
def group : MergeGroup := .operator 92682 92679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92682) (leftOrdinal := 0)
    (rightResult := 92679) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92686

namespace LeftMerge92695
def owner : Owner := ⟨.program ⟨257⟩, ⟨38997⟩⟩
def mergeEvent : Nat := 92695
def frameStart : Nat := 92603
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩] } }
def leftRaw : List Term := Proof.Events362.exact92691RawTerms
def rightRaw : List Term := Proof.Events361.exact92648RawTerms
def group : MergeGroup := .operator 92691 92648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92691) (leftOrdinal := 0)
    (rightResult := 92648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38994⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92695

namespace LeftMerge92696
def owner : Owner := ⟨.program ⟨257⟩, ⟨38997⟩⟩
def mergeEvent : Nat := 92696
def frameStart : Nat := 92603
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩] } }
def leftRaw : List Term := Proof.Events362.exact92691RawTerms
def rightRaw : List Term := Proof.Events361.exact92648RawTerms
def group : MergeGroup := .operator 92691 92648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92691) (leftOrdinal := 1)
    (rightResult := 92648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38994⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92696

namespace LeftMerge92698
def owner : Owner := ⟨.program ⟨257⟩, ⟨38997⟩⟩
def mergeEvent : Nat := 92698
def frameStart : Nat := 92603
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38459⟩⟩] } }
def rhsRaw : List Term := Proof.Events361.exact92645RawTerms
def group : MergeGroup := .relation 92697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92697) (rhsResult := 92645)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38994⟩⟩) ⟨38459⟩ 92645) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38459⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], [⟨.program ⟨257⟩, ⟨38459⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92698

namespace LeftMerge92706
def owner : Owner := ⟨.program ⟨257⟩, ⟨37470⟩⟩
def mergeEvent : Nat := 92706
def frameStart : Nat := 92603
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events361.exact92659RawTerms
def rightRaw : List Term := Proof.Events362.exact92702RawTerms
def group : MergeGroup := .operator 92659 92702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92659) (leftOrdinal := 0)
    (rightResult := 92702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37468⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92706

namespace LeftMerge92723
def owner : Owner := ⟨.program ⟨257⟩, ⟨37922⟩⟩
def mergeEvent : Nat := 92723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events362.exact92720RawTerms
def group : MergeGroup := .relation 92722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92722) (rhsResult := 92720)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 92721 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩) (none) 92720) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92723

namespace LeftMerge92724
def owner : Owner := ⟨.program ⟨257⟩, ⟨37922⟩⟩
def mergeEvent : Nat := 92724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩] } }
def rhsRaw : List Term := Proof.Events362.exact92720RawTerms
def group : MergeGroup := .relation 92722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92722) (rhsResult := 92720)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 92721 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩) (none) 92720) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38994⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92724

namespace LeftMerge92725
def owner : Owner := ⟨.program ⟨257⟩, ⟨37922⟩⟩
def mergeEvent : Nat := 92725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38459⟩⟩] } }
def rhsRaw : List Term := Proof.Events362.exact92720RawTerms
def group : MergeGroup := .relation 92722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92722) (rhsResult := 92720)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 92721 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩) (none) 92720) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38459⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], [⟨.program ⟨257⟩, ⟨38459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge92725

namespace LeftMerge92726
def owner : Owner := ⟨.program ⟨257⟩, ⟨37922⟩⟩
def mergeEvent : Nat := 92726
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events362.exact92720RawTerms
def group : MergeGroup := .relation 92722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 92722) (rhsResult := 92720)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 92721 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37919⟩⟩]⟩) (none) 92720) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37468⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37468⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92726

namespace LeftMerge92731
def owner : Owner := ⟨.program ⟨257⟩, ⟨38996⟩⟩
def mergeEvent : Nat := 92731
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38459⟩⟩] } }
def leftRaw : List Term := Proof.Events362.exact92727RawTerms
def rightRaw : List Term := Proof.Events361.exact92541RawTerms
def group : MergeGroup := .operator 92727 92541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 92727) (leftOrdinal := 2)
    (rightResult := 92541) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38459⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨13956⟩⟩, ⟨.program ⟨257⟩, ⟨37234⟩⟩], [⟨.program ⟨257⟩, ⟨38459⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge92731

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
