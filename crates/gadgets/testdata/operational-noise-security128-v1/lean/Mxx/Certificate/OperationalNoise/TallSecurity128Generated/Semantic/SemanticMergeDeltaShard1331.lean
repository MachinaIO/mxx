import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge217653
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217653
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 4)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217653

namespace LeftMerge217654
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217654
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 3)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217654

namespace LeftMerge217655
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217655
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 2)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217655

namespace LeftMerge217656
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217656
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 1)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217656

namespace LeftMerge217657
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217657
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 0)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217657

namespace LeftMerge217658
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217658
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 29)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217658

namespace LeftMerge217660
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217660
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48363⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217659) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48363⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217660

namespace LeftMerge217661
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217661
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 28)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217661

namespace LeftMerge217663
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217663
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45683⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217662
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217662) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45683⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217663

namespace LeftMerge217664
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217664
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 27)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217664

namespace LeftMerge217666
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217666
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42999⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217665) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42999⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217666

namespace LeftMerge217667
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217667
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 26)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217667

namespace LeftMerge217669
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217669
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40319⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217668) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40319⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217669

namespace LeftMerge217670
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217670
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 25)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217670

namespace LeftMerge217672
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217672
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37643⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }
def rhsRaw : List Term := Proof.Events849.exact217474RawTerms
def group : MergeGroup := .relation 217671
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 217671) (rhsResult := 217474)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71236⟩⟩) ⟨68830⟩ 217474) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68830⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37643⟩⟩], [⟨.program ⟨257⟩, ⟨68830⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217672

namespace LeftMerge217673
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217673
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 24)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge217673

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
