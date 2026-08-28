import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge271398
def owner : Owner := ⟨.program ⟨257⟩, ⟨56287⟩⟩
def mergeEvent : Nat := 271398
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events1060.exact271389RawTerms
def rightRaw : List Term := Proof.Events088.exact22621RawTerms
def group : MergeGroup := .operator 271389 22621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271389) (leftOrdinal := 0)
    (rightResult := 22621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271398

namespace LeftMerge271403
def owner : Owner := ⟨.program ⟨257⟩, ⟨56288⟩⟩
def mergeEvent : Nat := 271403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events1060.exact271399RawTerms
def rightRaw : List Term := Proof.Events1060.exact271369RawTerms
def group : MergeGroup := .operator 271399 271369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271399) (leftOrdinal := 1)
    (rightResult := 271369) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271403

namespace LeftMerge271411
def owner : Owner := ⟨.program ⟨257⟩, ⟨58389⟩⟩
def mergeEvent : Nat := 271411
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩] } }
def leftRaw : List Term := Proof.Events1060.exact271405RawTerms
def rightRaw : List Term := Proof.Events1059.exact271341RawTerms
def group : MergeGroup := .operator 271405 271341
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271405) (leftOrdinal := 1)
    (rightResult := 271341) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58388⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271411

namespace LeftMerge271413
def owner : Owner := ⟨.program ⟨257⟩, ⟨58389⟩⟩
def mergeEvent : Nat := 271413
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57919⟩⟩] } }
def rhsRaw : List Term := Proof.Events1059.exact271338RawTerms
def group : MergeGroup := .relation 271412
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271412) (rhsResult := 271338)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58388⟩⟩) ⟨57919⟩ 271338) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], [⟨.program ⟨257⟩, ⟨57919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271413

namespace LeftMerge271414
def owner : Owner := ⟨.program ⟨257⟩, ⟨58389⟩⟩
def mergeEvent : Nat := 271414
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩] } }
def leftRaw : List Term := Proof.Events1060.exact271405RawTerms
def rightRaw : List Term := Proof.Events1059.exact271341RawTerms
def group : MergeGroup := .operator 271405 271341
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271405) (leftOrdinal := 0)
    (rightResult := 271341) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58388⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271414

namespace LeftMerge271428
def owner : Owner := ⟨.program ⟨257⟩, ⟨57329⟩⟩
def mergeEvent : Nat := 271428
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57326⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1060.exact271422RawTerms
def group : MergeGroup := .operator 266120 271422
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 271422) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57326⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57326⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271428

namespace LeftMerge271507
def owner : Owner := ⟨.program ⟨257⟩, ⟨56281⟩⟩
def mergeEvent : Nat := 271507
def frameStart : Nat := 271477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1060.exact271503RawTerms
def rightRaw : List Term := Proof.Events1060.exact271500RawTerms
def group : MergeGroup := .operator 271503 271500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271503) (leftOrdinal := 0)
    (rightResult := 271500) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24910⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271507

namespace LeftMerge271537
def owner : Owner := ⟨.program ⟨257⟩, ⟨58216⟩⟩
def mergeEvent : Nat := 271537
def frameStart : Nat := 271477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1060.exact271533RawTerms
def rightRaw : List Term := Proof.Events1060.exact271531RawTerms
def group : MergeGroup := .operator 271533 271531
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271533) (leftOrdinal := 0)
    (rightResult := 271531) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271537

namespace LeftMerge271560
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def mergeEvent : Nat := 271560
def frameStart : Nat := 271477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events1060.exact271556RawTerms
def rightRaw : List Term := Proof.Events1060.exact271553RawTerms
def group : MergeGroup := .operator 271556 271553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271556) (leftOrdinal := 0)
    (rightResult := 271553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271560

namespace LeftMerge271569
def owner : Owner := ⟨.program ⟨257⟩, ⟨58391⟩⟩
def mergeEvent : Nat := 271569
def frameStart : Nat := 271477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩] } }
def leftRaw : List Term := Proof.Events1060.exact271565RawTerms
def rightRaw : List Term := Proof.Events1060.exact271522RawTerms
def group : MergeGroup := .operator 271565 271522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271565) (leftOrdinal := 0)
    (rightResult := 271522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58388⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271569

namespace LeftMerge271570
def owner : Owner := ⟨.program ⟨257⟩, ⟨58391⟩⟩
def mergeEvent : Nat := 271570
def frameStart : Nat := 271477
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩] } }
def leftRaw : List Term := Proof.Events1060.exact271565RawTerms
def rightRaw : List Term := Proof.Events1060.exact271522RawTerms
def group : MergeGroup := .operator 271565 271522
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271565) (leftOrdinal := 1)
    (rightResult := 271522) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58388⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271570

namespace LeftMerge271572
def owner : Owner := ⟨.program ⟨257⟩, ⟨58391⟩⟩
def mergeEvent : Nat := 271572
def frameStart : Nat := 271477
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57919⟩⟩] } }
def rhsRaw : List Term := Proof.Events1060.exact271519RawTerms
def group : MergeGroup := .relation 271571
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271571) (rhsResult := 271519)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58388⟩⟩) ⟨57919⟩ 271519) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], [⟨.program ⟨257⟩, ⟨57919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271572

namespace LeftMerge271580
def owner : Owner := ⟨.program ⟨257⟩, ⟨56784⟩⟩
def mergeEvent : Nat := 271580
def frameStart : Nat := 271477
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56782⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1060.exact271533RawTerms
def rightRaw : List Term := Proof.Events1060.exact271576RawTerms
def group : MergeGroup := .operator 271533 271576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271533) (leftOrdinal := 0)
    (rightResult := 271576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56782⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56782⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271580

namespace LeftMerge271597
def owner : Owner := ⟨.program ⟨257⟩, ⟨57329⟩⟩
def mergeEvent : Nat := 271597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events1060.exact271594RawTerms
def group : MergeGroup := .relation 271596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271596) (rhsResult := 271594)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57326⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 271595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57326⟩⟩]⟩) (none) 271594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271597

namespace LeftMerge271598
def owner : Owner := ⟨.program ⟨257⟩, ⟨57329⟩⟩
def mergeEvent : Nat := 271598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩] } }
def rhsRaw : List Term := Proof.Events1060.exact271594RawTerms
def group : MergeGroup := .relation 271596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271596) (rhsResult := 271594)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57326⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 271595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57326⟩⟩]⟩) (none) 271594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58388⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge271598

namespace LeftMerge271599
def owner : Owner := ⟨.program ⟨257⟩, ⟨57329⟩⟩
def mergeEvent : Nat := 271599
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57919⟩⟩] } }
def rhsRaw : List Term := Proof.Events1060.exact271594RawTerms
def group : MergeGroup := .relation 271596
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 271596) (rhsResult := 271594)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57326⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 271595 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57326⟩⟩]⟩) (none) 271594) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24910⟩⟩, ⟨.program ⟨257⟩, ⟨56280⟩⟩], [⟨.program ⟨257⟩, ⟨57919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271599

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
