import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge217507
def owner : Owner := ⟨.program ⟨257⟩, ⟨69089⟩⟩
def mergeEvent : Nat := 217507
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events849.exact217488RawTerms
def rightRaw : List Term := Proof.Events849.exact217486RawTerms
def group : MergeGroup := .operator 217488 217486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217488) (leftOrdinal := 0)
    (rightResult := 217486) (rightOrdinal := 2) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22086⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22086⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217507

namespace LeftMerge217508
def owner : Owner := ⟨.program ⟨257⟩, ⟨69089⟩⟩
def mergeEvent : Nat := 217508
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events849.exact217488RawTerms
def rightRaw : List Term := Proof.Events849.exact217486RawTerms
def group : MergeGroup := .operator 217488 217486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217488) (leftOrdinal := 0)
    (rightResult := 217486) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18866⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18866⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217508

namespace LeftMerge217509
def owner : Owner := ⟨.program ⟨257⟩, ⟨69089⟩⟩
def mergeEvent : Nat := 217509
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events849.exact217488RawTerms
def rightRaw : List Term := Proof.Events849.exact217486RawTerms
def group : MergeGroup := .operator 217488 217486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217488) (leftOrdinal := 0)
    (rightResult := 217486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16035⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16035⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217509

namespace LeftMerge217640
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217640
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 17)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217640

namespace LeftMerge217641
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217641
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 16)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217641

namespace LeftMerge217642
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217642
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 15)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217642

namespace LeftMerge217643
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217643
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 14)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217643

namespace LeftMerge217644
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217644
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 13)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217644

namespace LeftMerge217645
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217645
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 12)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217645

namespace LeftMerge217646
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217646
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 11)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217646

namespace LeftMerge217647
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217647
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 10)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217647

namespace LeftMerge217648
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217648
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 9)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217648

namespace LeftMerge217649
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217649
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 8)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217649

namespace LeftMerge217650
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217650
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 7)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217650

namespace LeftMerge217651
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217651
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 6)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217651

namespace LeftMerge217652
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def mergeEvent : Nat := 217652
def frameStart : Nat := 216961
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩] } }
def leftRaw : List Term := Proof.Events850.exact217636RawTerms
def rightRaw : List Term := Proof.Events849.exact217477RawTerms
def group : MergeGroup := .operator 217636 217477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 217636) (leftOrdinal := 5)
    (rightResult := 217477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71236⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71236⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge217652

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
