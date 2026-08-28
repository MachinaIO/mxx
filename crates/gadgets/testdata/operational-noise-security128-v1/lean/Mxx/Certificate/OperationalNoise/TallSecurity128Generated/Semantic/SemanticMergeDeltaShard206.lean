import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge37398
def owner : Owner := ⟨.program ⟨257⟩, ⟨56755⟩⟩
def mergeEvent : Nat := 37398
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events146.exact37389RawTerms
def rightRaw : List Term := Proof.Events088.exact22621RawTerms
def group : MergeGroup := .operator 37389 22621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37389) (leftOrdinal := 0)
    (rightResult := 22621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37398

namespace LeftMerge37403
def owner : Owner := ⟨.program ⟨257⟩, ⟨56756⟩⟩
def mergeEvent : Nat := 37403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events146.exact37399RawTerms
def rightRaw : List Term := Proof.Events145.exact37369RawTerms
def group : MergeGroup := .operator 37399 37369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37399) (leftOrdinal := 1)
    (rightResult := 37369) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37403

namespace LeftMerge37411
def owner : Owner := ⟨.program ⟨257⟩, ⟨58579⟩⟩
def mergeEvent : Nat := 37411
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩] } }
def leftRaw : List Term := Proof.Events146.exact37405RawTerms
def rightRaw : List Term := Proof.Events145.exact37341RawTerms
def group : MergeGroup := .operator 37405 37341
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37405) (leftOrdinal := 1)
    (rightResult := 37341) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58578⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37411

namespace LeftMerge37413
def owner : Owner := ⟨.program ⟨257⟩, ⟨58579⟩⟩
def mergeEvent : Nat := 37413
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58023⟩⟩] } }
def rhsRaw : List Term := Proof.Events145.exact37338RawTerms
def group : MergeGroup := .relation 37412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37412) (rhsResult := 37338)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58578⟩⟩) ⟨58023⟩ 37338) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58023⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], [⟨.program ⟨257⟩, ⟨58023⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37413

namespace LeftMerge37414
def owner : Owner := ⟨.program ⟨257⟩, ⟨58579⟩⟩
def mergeEvent : Nat := 37414
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩] } }
def leftRaw : List Term := Proof.Events146.exact37405RawTerms
def rightRaw : List Term := Proof.Events145.exact37341RawTerms
def group : MergeGroup := .operator 37405 37341
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37405) (leftOrdinal := 0)
    (rightResult := 37341) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58578⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37414

namespace LeftMerge37428
def owner : Owner := ⟨.program ⟨257⟩, ⟨57502⟩⟩
def mergeEvent : Nat := 37428
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57499⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events146.exact37422RawTerms
def group : MergeGroup := .operator 32120 37422
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 37422) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57499⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37428

namespace LeftMerge37507
def owner : Owner := ⟨.program ⟨257⟩, ⟨56749⟩⟩
def mergeEvent : Nat := 37507
def frameStart : Nat := 37477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events146.exact37503RawTerms
def rightRaw : List Term := Proof.Events146.exact37500RawTerms
def group : MergeGroup := .operator 37503 37500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37503) (leftOrdinal := 0)
    (rightResult := 37500) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25118⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37507

namespace LeftMerge37537
def owner : Owner := ⟨.program ⟨257⟩, ⟨58284⟩⟩
def mergeEvent : Nat := 37537
def frameStart : Nat := 37477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events146.exact37533RawTerms
def rightRaw : List Term := Proof.Events146.exact37531RawTerms
def group : MergeGroup := .operator 37533 37531
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37533) (leftOrdinal := 0)
    (rightResult := 37531) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37537

namespace LeftMerge37560
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 37560
def frameStart : Nat := 37477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events146.exact37556RawTerms
def rightRaw : List Term := Proof.Events146.exact37553RawTerms
def group : MergeGroup := .operator 37556 37553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37556) (leftOrdinal := 0)
    (rightResult := 37553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37560

namespace LeftMerge37569
def owner : Owner := ⟨.program ⟨257⟩, ⟨58581⟩⟩
def mergeEvent : Nat := 37569
def frameStart : Nat := 37477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩] } }
def leftRaw : List Term := Proof.Events146.exact37565RawTerms
def rightRaw : List Term := Proof.Events146.exact37522RawTerms
def group : MergeGroup := .operator 37565 37522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37565) (leftOrdinal := 0)
    (rightResult := 37522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58578⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37569

namespace LeftMerge37570
def owner : Owner := ⟨.program ⟨257⟩, ⟨58581⟩⟩
def mergeEvent : Nat := 37570
def frameStart : Nat := 37477
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩] } }
def leftRaw : List Term := Proof.Events146.exact37565RawTerms
def rightRaw : List Term := Proof.Events146.exact37522RawTerms
def group : MergeGroup := .operator 37565 37522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37565) (leftOrdinal := 1)
    (rightResult := 37522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58578⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37570

namespace LeftMerge37572
def owner : Owner := ⟨.program ⟨257⟩, ⟨58581⟩⟩
def mergeEvent : Nat := 37572
def frameStart : Nat := 37477
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58023⟩⟩] } }
def rhsRaw : List Term := Proof.Events146.exact37519RawTerms
def group : MergeGroup := .relation 37571
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37571) (rhsResult := 37519)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58578⟩⟩) ⟨58023⟩ 37519) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58023⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], [⟨.program ⟨257⟩, ⟨58023⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37572

namespace LeftMerge37580
def owner : Owner := ⟨.program ⟨257⟩, ⟨56922⟩⟩
def mergeEvent : Nat := 37580
def frameStart : Nat := 37477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events146.exact37533RawTerms
def rightRaw : List Term := Proof.Events146.exact37576RawTerms
def group : MergeGroup := .operator 37533 37576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 37533) (leftOrdinal := 0)
    (rightResult := 37576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56920⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37580

namespace LeftMerge37597
def owner : Owner := ⟨.program ⟨257⟩, ⟨57502⟩⟩
def mergeEvent : Nat := 37597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events146.exact37594RawTerms
def group : MergeGroup := .relation 37596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37596) (rhsResult := 37594)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57499⟩⟩]⟩) (none) 37594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37597

namespace LeftMerge37598
def owner : Owner := ⟨.program ⟨257⟩, ⟨57502⟩⟩
def mergeEvent : Nat := 37598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩] } }
def rhsRaw : List Term := Proof.Events146.exact37594RawTerms
def group : MergeGroup := .relation 37596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37596) (rhsResult := 37594)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57499⟩⟩]⟩) (none) 37594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58578⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge37598

namespace LeftMerge37599
def owner : Owner := ⟨.program ⟨257⟩, ⟨57502⟩⟩
def mergeEvent : Nat := 37599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58023⟩⟩] } }
def rhsRaw : List Term := Proof.Events146.exact37594RawTerms
def group : MergeGroup := .relation 37596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 37596) (rhsResult := 37594)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 37595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57499⟩⟩]⟩) (none) 37594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58023⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25118⟩⟩, ⟨.program ⟨257⟩, ⟨56748⟩⟩], [⟨.program ⟨257⟩, ⟨58023⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge37599

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
