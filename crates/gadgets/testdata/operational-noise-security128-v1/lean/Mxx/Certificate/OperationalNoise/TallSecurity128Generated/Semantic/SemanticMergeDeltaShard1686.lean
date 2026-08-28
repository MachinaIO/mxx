import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge273488
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 273488
def frameStart : Nat := 273405
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events1068.exact273484RawTerms
def rightRaw : List Term := Proof.Events1068.exact273481RawTerms
def group : MergeGroup := .operator 273484 273481
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273484) (leftOrdinal := 0)
    (rightResult := 273481) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273488

namespace LeftMerge273497
def owner : Owner := ⟨.program ⟨257⟩, ⟨23351⟩⟩
def mergeEvent : Nat := 273497
def frameStart : Nat := 273405
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩] } }
def leftRaw : List Term := Proof.Events1068.exact273493RawTerms
def rightRaw : List Term := Proof.Events1068.exact273450RawTerms
def group : MergeGroup := .operator 273493 273450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273493) (leftOrdinal := 0)
    (rightResult := 273450) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23348⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273497

namespace LeftMerge273498
def owner : Owner := ⟨.program ⟨257⟩, ⟨23351⟩⟩
def mergeEvent : Nat := 273498
def frameStart : Nat := 273405
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩] } }
def leftRaw : List Term := Proof.Events1068.exact273493RawTerms
def rightRaw : List Term := Proof.Events1068.exact273450RawTerms
def group : MergeGroup := .operator 273493 273450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273493) (leftOrdinal := 1)
    (rightResult := 273450) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23348⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273498

namespace LeftMerge273500
def owner : Owner := ⟨.program ⟨257⟩, ⟨23351⟩⟩
def mergeEvent : Nat := 273500
def frameStart : Nat := 273405
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22879⟩⟩] } }
def rhsRaw : List Term := Proof.Events1068.exact273447RawTerms
def group : MergeGroup := .relation 273499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273499) (rhsResult := 273447)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23348⟩⟩) ⟨22879⟩ 273447) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22879⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], [⟨.program ⟨257⟩, ⟨22879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273500

namespace LeftMerge273508
def owner : Owner := ⟨.program ⟨257⟩, ⟨21744⟩⟩
def mergeEvent : Nat := 273508
def frameStart : Nat := 273405
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1068.exact273461RawTerms
def rightRaw : List Term := Proof.Events1068.exact273504RawTerms
def group : MergeGroup := .operator 273461 273504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273461) (leftOrdinal := 0)
    (rightResult := 273504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273508

namespace LeftMerge273525
def owner : Owner := ⟨.program ⟨257⟩, ⟨22289⟩⟩
def mergeEvent : Nat := 273525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events1068.exact273522RawTerms
def group : MergeGroup := .relation 273524
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273524) (rhsResult := 273522)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22286⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 273523 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22286⟩⟩]⟩) (none) 273522) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273525

namespace LeftMerge273526
def owner : Owner := ⟨.program ⟨257⟩, ⟨22289⟩⟩
def mergeEvent : Nat := 273526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩] } }
def rhsRaw : List Term := Proof.Events1068.exact273522RawTerms
def group : MergeGroup := .relation 273524
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273524) (rhsResult := 273522)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22286⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 273523 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22286⟩⟩]⟩) (none) 273522) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273526

namespace LeftMerge273527
def owner : Owner := ⟨.program ⟨257⟩, ⟨22289⟩⟩
def mergeEvent : Nat := 273527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22879⟩⟩] } }
def rhsRaw : List Term := Proof.Events1068.exact273522RawTerms
def group : MergeGroup := .relation 273524
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273524) (rhsResult := 273522)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22286⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 273523 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22286⟩⟩]⟩) (none) 273522) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22879⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], [⟨.program ⟨257⟩, ⟨22879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273527

namespace LeftMerge273528
def owner : Owner := ⟨.program ⟨257⟩, ⟨22289⟩⟩
def mergeEvent : Nat := 273528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1068.exact273522RawTerms
def group : MergeGroup := .relation 273524
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273524) (rhsResult := 273522)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22286⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 273523 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22286⟩⟩]⟩) (none) 273522) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273528

namespace LeftMerge273533
def owner : Owner := ⟨.program ⟨257⟩, ⟨23350⟩⟩
def mergeEvent : Nat := 273533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22879⟩⟩] } }
def leftRaw : List Term := Proof.Events1068.exact273529RawTerms
def rightRaw : List Term := Proof.Events1067.exact273343RawTerms
def group : MergeGroup := .operator 273529 273343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273529) (leftOrdinal := 2)
    (rightResult := 273343) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22879⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22879⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨20976⟩⟩, ⟨.program ⟨257⟩, ⟨21294⟩⟩], [⟨.program ⟨257⟩, ⟨22879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273533

namespace LeftMerge273534
def owner : Owner := ⟨.program ⟨257⟩, ⟨23350⟩⟩
def mergeEvent : Nat := 273534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩] } }
def leftRaw : List Term := Proof.Events1068.exact273529RawTerms
def rightRaw : List Term := Proof.Events1067.exact273343RawTerms
def group : MergeGroup := .operator 273529 273343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273529) (leftOrdinal := 1)
    (rightResult := 273343) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23348⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273534

namespace LeftMerge273542
def owner : Owner := ⟨.program ⟨257⟩, ⟨23617⟩⟩
def mergeEvent : Nat := 273542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23615⟩⟩] } }
def leftRaw : List Term := Proof.Events1068.exact273536RawTerms
def rightRaw : List Term := Proof.Events1067.exact273259RawTerms
def group : MergeGroup := .operator 273536 273259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273536) (leftOrdinal := 0)
    (rightResult := 273259) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23615⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273542

namespace LeftMerge273543
def owner : Owner := ⟨.program ⟨257⟩, ⟨23617⟩⟩
def mergeEvent : Nat := 273543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23615⟩⟩] } }
def leftRaw : List Term := Proof.Events1068.exact273536RawTerms
def rightRaw : List Term := Proof.Events1067.exact273259RawTerms
def group : MergeGroup := .operator 273536 273259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273536) (leftOrdinal := 1)
    (rightResult := 273259) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23615⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273543

namespace LeftMerge273545
def owner : Owner := ⟨.program ⟨257⟩, ⟨23617⟩⟩
def mergeEvent : Nat := 273545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23006⟩⟩] } }
def rhsRaw : List Term := Proof.Events1067.exact273256RawTerms
def group : MergeGroup := .relation 273544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273544) (rhsResult := 273256)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23615⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23615⟩⟩) ⟨23006⟩ 273256) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23006⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨21742⟩⟩], [⟨.program ⟨257⟩, ⟨23006⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273545

namespace LeftMerge273559
def owner : Owner := ⟨.program ⟨257⟩, ⟨22513⟩⟩
def mergeEvent : Nat := 273559
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22510⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1068.exact273553RawTerms
def group : MergeGroup := .operator 266120 273553
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 273553) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22510⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22510⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273559

namespace LeftMerge273680
def owner : Owner := ⟨.program ⟨257⟩, ⟨23256⟩⟩
def mergeEvent : Nat := 273680
def frameStart : Nat := 273614
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21742⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1069.exact273676RawTerms
def rightRaw : List Term := Proof.Events1069.exact273674RawTerms
def group : MergeGroup := .operator 273676 273674
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273676) (leftOrdinal := 0)
    (rightResult := 273674) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21742⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21742⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273680

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
