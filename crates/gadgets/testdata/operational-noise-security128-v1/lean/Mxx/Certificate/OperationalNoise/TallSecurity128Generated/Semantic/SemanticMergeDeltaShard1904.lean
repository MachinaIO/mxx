import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge307472
def owner : Owner := ⟨.program ⟨257⟩, ⟨23556⟩⟩
def mergeEvent : Nat := 307472
def frameStart : Nat := 307405
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩] } }
def leftRaw : List Term := Proof.Events1201.exact307467RawTerms
def rightRaw : List Term := Proof.Events1200.exact307444RawTerms
def group : MergeGroup := .operator 307467 307444
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 307467) (leftOrdinal := 1)
    (rightResult := 307444) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23555⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge307472

namespace LeftMerge307474
def owner : Owner := ⟨.program ⟨257⟩, ⟨23556⟩⟩
def mergeEvent : Nat := 307474
def frameStart : Nat := 307405
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22990⟩⟩] } }
def rhsRaw : List Term := Proof.Events1200.exact307441RawTerms
def group : MergeGroup := .relation 307473
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 307473) (rhsResult := 307441)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23555⟩⟩) ⟨22990⟩ 307441) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22990⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨22990⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge307474

namespace LeftMerge307482
def owner : Owner := ⟨.program ⟨257⟩, ⟨21894⟩⟩
def mergeEvent : Nat := 307482
def frameStart : Nat := 307405
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21891⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1200.exact307455RawTerms
def rightRaw : List Term := Proof.Events1201.exact307478RawTerms
def group : MergeGroup := .operator 307455 307478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 307455) (leftOrdinal := 0)
    (rightResult := 307478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21891⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge307482

namespace LeftMerge307499
def owner : Owner := ⟨.program ⟨257⟩, ⟨22475⟩⟩
def mergeEvent : Nat := 307499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩] } }
def rhsRaw : List Term := Proof.Events1201.exact307496RawTerms
def group : MergeGroup := .relation 307498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 307498) (rhsResult := 307496)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22472⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 307497 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22472⟩⟩]⟩) (none) 307496) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge307499

namespace LeftMerge307500
def owner : Owner := ⟨.program ⟨257⟩, ⟨22475⟩⟩
def mergeEvent : Nat := 307500
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩] } }
def rhsRaw : List Term := Proof.Events1201.exact307496RawTerms
def group : MergeGroup := .relation 307498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 307498) (rhsResult := 307496)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22472⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 307497 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22472⟩⟩]⟩) (none) 307496) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge307500

namespace LeftMerge307501
def owner : Owner := ⟨.program ⟨257⟩, ⟨22475⟩⟩
def mergeEvent : Nat := 307501
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22990⟩⟩] } }
def rhsRaw : List Term := Proof.Events1201.exact307496RawTerms
def group : MergeGroup := .relation 307498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 307498) (rhsResult := 307496)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22472⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 307497 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22472⟩⟩]⟩) (none) 307496) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22990⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨22990⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge307501

namespace LeftMerge307502
def owner : Owner := ⟨.program ⟨257⟩, ⟨22475⟩⟩
def mergeEvent : Nat := 307502
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1201.exact307496RawTerms
def group : MergeGroup := .relation 307498
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 307498) (rhsResult := 307496)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22472⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 307497 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22472⟩⟩]⟩) (none) 307496) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21891⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge307502

namespace LeftMerge307507
def owner : Owner := ⟨.program ⟨257⟩, ⟨23558⟩⟩
def mergeEvent : Nat := 307507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩] } }
def leftRaw : List Term := Proof.Events1201.exact307503RawTerms
def rightRaw : List Term := Proof.Events1200.exact307349RawTerms
def group : MergeGroup := .operator 307503 307349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 307503) (leftOrdinal := 0)
    (rightResult := 307349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23555⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge307507

namespace LeftMerge307508
def owner : Owner := ⟨.program ⟨257⟩, ⟨23558⟩⟩
def mergeEvent : Nat := 307508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22990⟩⟩] } }
def leftRaw : List Term := Proof.Events1201.exact307503RawTerms
def rightRaw : List Term := Proof.Events1200.exact307349RawTerms
def group : MergeGroup := .operator 307503 307349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 307503) (leftOrdinal := 2)
    (rightResult := 307349) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22990⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22990⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21728⟩⟩], [⟨.program ⟨257⟩, ⟨22990⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge307508

namespace LeftMerge307516
def owner : Owner := ⟨.program ⟨257⟩, ⟨23559⟩⟩
def mergeEvent : Nat := 307516
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩] } }
def leftRaw : List Term := Proof.Events1201.exact307510RawTerms
def rightRaw : List Term := Proof.Events061.exact15842RawTerms
def group : MergeGroup := .operator 307510 15842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 307510) (leftOrdinal := 0)
    (rightResult := 15842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7155⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge307516

namespace LeftMerge307517
def owner : Owner := ⟨.program ⟨257⟩, ⟨23559⟩⟩
def mergeEvent : Nat := 307517
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩] } }
def leftRaw : List Term := Proof.Events1201.exact307510RawTerms
def rightRaw : List Term := Proof.Events061.exact15842RawTerms
def group : MergeGroup := .operator 307510 15842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 307510) (leftOrdinal := 1)
    (rightResult := 15842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7155⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge307517

namespace LeftMerge307519
def owner : Owner := ⟨.program ⟨257⟩, ⟨23559⟩⟩
def mergeEvent : Nat := 307519
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15835RawTerms
def group : MergeGroup := .relation 307518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 307518) (rhsResult := 15835)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7155⟩⟩) ⟨7043⟩ 15835) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge307519

namespace LeftMerge307533
def owner : Owner := ⟨.program ⟨257⟩, ⟨20337⟩⟩
def mergeEvent : Nat := 307533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20335⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302301RawTerms
def rightRaw : List Term := Proof.Events1201.exact307527RawTerms
def group : MergeGroup := .operator 302301 307527
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302301) (leftOrdinal := 0)
    (rightResult := 307527) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20335⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20335⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge307533

namespace LeftMerge307534
def owner : Owner := ⟨.program ⟨257⟩, ⟨20337⟩⟩
def mergeEvent : Nat := 307534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20335⟩⟩] } }
def leftRaw : List Term := Proof.Events1180.exact302301RawTerms
def rightRaw : List Term := Proof.Events1201.exact307527RawTerms
def group : MergeGroup := .operator 302301 307527
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302301) (leftOrdinal := 1)
    (rightResult := 307527) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20335⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20335⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge307534

namespace LeftMerge307536
def owner : Owner := ⟨.program ⟨257⟩, ⟨20337⟩⟩
def mergeEvent : Nat := 307536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18508⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1201.exact307524RawTerms
def group : MergeGroup := .relation 307535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 307535) (rhsResult := 307524)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18508⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20335⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20335⟩⟩) ⟨19770⟩ 307524) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18508⟩⟩], [⟨.program ⟨257⟩, ⟨19770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge307536

namespace LeftMerge307550
def owner : Owner := ⟨.program ⟨257⟩, ⟨19255⟩⟩
def mergeEvent : Nat := 307550
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19252⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1201.exact307544RawTerms
def group : MergeGroup := .operator 295195 307544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 307544) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19252⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19252⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge307550

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
