import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge227444
def owner : Owner := ⟨.program ⟨257⟩, ⟨61864⟩⟩
def mergeEvent : Nat := 227444
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61092⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227439RawTerms
def rightRaw : List Term := Proof.Events887.exact227261RawTerms
def group : MergeGroup := .operator 227439 227261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227439) (leftOrdinal := 2)
    (rightResult := 227261) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61092⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61092⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨59820⟩⟩], [⟨.program ⟨257⟩, ⟨61092⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227444

namespace LeftMerge227470
def owner : Owner := ⟨.program ⟨257⟩, ⟨24999⟩⟩
def mergeEvent : Nat := 227470
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10819RawTerms
def rightRaw : List Term := Proof.Events867.exact222153RawTerms
def group : MergeGroup := .operator 10819 222153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10819) (leftOrdinal := 0)
    (rightResult := 222153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24998⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227470

namespace LeftMerge227475
def owner : Owner := ⟨.program ⟨257⟩, ⟨8465⟩⟩
def mergeEvent : Nat := 227475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222023RawTerms
def rightRaw : List Term := Proof.Events088.exact22591RawTerms
def group : MergeGroup := .operator 222023 22591
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222023) (leftOrdinal := 0)
    (rightResult := 22591) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227475

namespace LeftMerge227492
def owner : Owner := ⟨.program ⟨257⟩, ⟨56481⟩⟩
def mergeEvent : Nat := 227492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227486RawTerms
def rightRaw : List Term := Proof.Events042.exact10822RawTerms
def group : MergeGroup := .operator 227486 10822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227486) (leftOrdinal := 1)
    (rightResult := 10822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227492

namespace LeftMerge227493
def owner : Owner := ⟨.program ⟨257⟩, ⟨56481⟩⟩
def mergeEvent : Nat := 227493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227486RawTerms
def rightRaw : List Term := Proof.Events042.exact10822RawTerms
def group : MergeGroup := .operator 227486 10822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227486) (leftOrdinal := 0)
    (rightResult := 10822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227493

namespace LeftMerge227498
def owner : Owner := ⟨.program ⟨257⟩, ⟨56482⟩⟩
def mergeEvent : Nat := 227498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events042.exact10822RawTerms
def rightRaw : List Term := Proof.Events867.exact222153RawTerms
def group : MergeGroup := .operator 10822 222153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10822) (leftOrdinal := 0)
    (rightResult := 222153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227498

namespace LeftMerge227503
def owner : Owner := ⟨.program ⟨257⟩, ⟨8482⟩⟩
def mergeEvent : Nat := 227503
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }
def leftRaw : List Term := Proof.Events867.exact222023RawTerms
def rightRaw : List Term := Proof.Events088.exact22632RawTerms
def group : MergeGroup := .operator 222023 22632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222023) (leftOrdinal := 0)
    (rightResult := 22632) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227503

namespace LeftMerge227520
def owner : Owner := ⟨.program ⟨257⟩, ⟨56485⟩⟩
def mergeEvent : Nat := 227520
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227514RawTerms
def rightRaw : List Term := Proof.Events088.exact22621RawTerms
def group : MergeGroup := .operator 227514 22621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227514) (leftOrdinal := 1)
    (rightResult := 22621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227520

namespace LeftMerge227522
def owner : Owner := ⟨.program ⟨257⟩, ⟨56485⟩⟩
def mergeEvent : Nat := 227522
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def rhsRaw : List Term := Proof.Events088.exact22591RawTerms
def group : MergeGroup := .relation 227521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227521) (rhsResult := 22591)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9532⟩⟩) ⟨7273⟩ 22591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227522

namespace LeftMerge227523
def owner : Owner := ⟨.program ⟨257⟩, ⟨56485⟩⟩
def mergeEvent : Nat := 227523
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227514RawTerms
def rightRaw : List Term := Proof.Events088.exact22621RawTerms
def group : MergeGroup := .operator 227514 22621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227514) (leftOrdinal := 0)
    (rightResult := 22621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227523

namespace LeftMerge227528
def owner : Owner := ⟨.program ⟨257⟩, ⟨56486⟩⟩
def mergeEvent : Nat := 227528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227524RawTerms
def rightRaw : List Term := Proof.Events888.exact227494RawTerms
def group : MergeGroup := .operator 227524 227494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227524) (leftOrdinal := 1)
    (rightResult := 227494) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨7273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227528

namespace LeftMerge227536
def owner : Owner := ⟨.program ⟨257⟩, ⟨58469⟩⟩
def mergeEvent : Nat := 227536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58468⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227530RawTerms
def rightRaw : List Term := Proof.Events888.exact227466RawTerms
def group : MergeGroup := .operator 227530 227466
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227530) (leftOrdinal := 1)
    (rightResult := 227466) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58468⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58468⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227536

namespace LeftMerge227538
def owner : Owner := ⟨.program ⟨257⟩, ⟨58469⟩⟩
def mergeEvent : Nat := 227538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57963⟩⟩] } }
def rhsRaw : List Term := Proof.Events888.exact227463RawTerms
def group : MergeGroup := .relation 227537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 227537) (rhsResult := 227463)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58468⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58468⟩⟩) ⟨57963⟩ 227463) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57963⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], [⟨.program ⟨257⟩, ⟨57963⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge227538

namespace LeftMerge227539
def owner : Owner := ⟨.program ⟨257⟩, ⟨58469⟩⟩
def mergeEvent : Nat := 227539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58468⟩⟩] } }
def leftRaw : List Term := Proof.Events888.exact227530RawTerms
def rightRaw : List Term := Proof.Events888.exact227466RawTerms
def group : MergeGroup := .operator 227530 227466
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227530) (leftOrdinal := 0)
    (rightResult := 227466) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58468⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58468⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227539

namespace LeftMerge227553
def owner : Owner := ⟨.program ⟨257⟩, ⟨57402⟩⟩
def mergeEvent : Nat := 227553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57399⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events888.exact227547RawTerms
def group : MergeGroup := .operator 222245 227547
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 227547) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57399⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57399⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227553

namespace LeftMerge227632
def owner : Owner := ⟨.program ⟨257⟩, ⟨56479⟩⟩
def mergeEvent : Nat := 227632
def frameStart : Nat := 227602
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events889.exact227628RawTerms
def rightRaw : List Term := Proof.Events889.exact227625RawTerms
def group : MergeGroup := .operator 227628 227625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 227628) (leftOrdinal := 0)
    (rightResult := 227625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56478⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24998⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24998⟩⟩, ⟨.program ⟨257⟩, ⟨56478⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge227632

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
