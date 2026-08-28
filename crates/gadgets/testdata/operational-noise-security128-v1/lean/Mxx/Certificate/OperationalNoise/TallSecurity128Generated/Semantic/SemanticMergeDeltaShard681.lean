import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge113430
def owner : Owner := ⟨.program ⟨257⟩, ⟨17371⟩⟩
def mergeEvent : Nat := 113430
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16855⟩⟩] } }
def rhsRaw : List Term := Proof.Events442.exact113355RawTerms
def group : MergeGroup := .relation 113429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113429) (rhsResult := 113355)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17370⟩⟩) ⟨16855⟩ 113355) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16855⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], [⟨.program ⟨257⟩, ⟨16855⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113430

namespace LeftMerge113431
def owner : Owner := ⟨.program ⟨257⟩, ⟨17371⟩⟩
def mergeEvent : Nat := 113431
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩] } }
def leftRaw : List Term := Proof.Events443.exact113422RawTerms
def rightRaw : List Term := Proof.Events442.exact113358RawTerms
def group : MergeGroup := .operator 113422 113358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113422) (leftOrdinal := 0)
    (rightResult := 113358) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17370⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113431

namespace LeftMerge113445
def owner : Owner := ⟨.program ⟨257⟩, ⟨16302⟩⟩
def mergeEvent : Nat := 113445
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events443.exact113439RawTerms
def group : MergeGroup := .operator 105245 113439
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 113439) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16299⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113445

namespace LeftMerge113524
def owner : Owner := ⟨.program ⟨257⟩, ⟨15499⟩⟩
def mergeEvent : Nat := 113524
def frameStart : Nat := 113494
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events443.exact113520RawTerms
def rightRaw : List Term := Proof.Events443.exact113517RawTerms
def group : MergeGroup := .operator 113520 113517
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113520) (leftOrdinal := 0)
    (rightResult := 113517) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12396⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113524

namespace LeftMerge113554
def owner : Owner := ⟨.program ⟨257⟩, ⟨17132⟩⟩
def mergeEvent : Nat := 113554
def frameStart : Nat := 113494
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events443.exact113550RawTerms
def rightRaw : List Term := Proof.Events443.exact113548RawTerms
def group : MergeGroup := .operator 113550 113548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113550) (leftOrdinal := 0)
    (rightResult := 113548) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113554

namespace LeftMerge113577
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 113577
def frameStart : Nat := 113494
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events443.exact113573RawTerms
def rightRaw : List Term := Proof.Events443.exact113570RawTerms
def group : MergeGroup := .operator 113573 113570
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113573) (leftOrdinal := 0)
    (rightResult := 113570) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113577

namespace LeftMerge113586
def owner : Owner := ⟨.program ⟨257⟩, ⟨17373⟩⟩
def mergeEvent : Nat := 113586
def frameStart : Nat := 113494
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩] } }
def leftRaw : List Term := Proof.Events443.exact113582RawTerms
def rightRaw : List Term := Proof.Events443.exact113539RawTerms
def group : MergeGroup := .operator 113582 113539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113582) (leftOrdinal := 0)
    (rightResult := 113539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17370⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113586

namespace LeftMerge113587
def owner : Owner := ⟨.program ⟨257⟩, ⟨17373⟩⟩
def mergeEvent : Nat := 113587
def frameStart : Nat := 113494
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩] } }
def leftRaw : List Term := Proof.Events443.exact113582RawTerms
def rightRaw : List Term := Proof.Events443.exact113539RawTerms
def group : MergeGroup := .operator 113582 113539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113582) (leftOrdinal := 1)
    (rightResult := 113539) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17370⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113587

namespace LeftMerge113589
def owner : Owner := ⟨.program ⟨257⟩, ⟨17373⟩⟩
def mergeEvent : Nat := 113589
def frameStart : Nat := 113494
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16855⟩⟩] } }
def rhsRaw : List Term := Proof.Events443.exact113536RawTerms
def group : MergeGroup := .relation 113588
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113588) (rhsResult := 113536)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17370⟩⟩) ⟨16855⟩ 113536) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16855⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], [⟨.program ⟨257⟩, ⟨16855⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113589

namespace LeftMerge113597
def owner : Owner := ⟨.program ⟨257⟩, ⟨15798⟩⟩
def mergeEvent : Nat := 113597
def frameStart : Nat := 113494
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events443.exact113550RawTerms
def rightRaw : List Term := Proof.Events443.exact113593RawTerms
def group : MergeGroup := .operator 113550 113593
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113550) (leftOrdinal := 0)
    (rightResult := 113593) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15796⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113597

namespace LeftMerge113614
def owner : Owner := ⟨.program ⟨257⟩, ⟨16302⟩⟩
def mergeEvent : Nat := 113614
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events443.exact113611RawTerms
def group : MergeGroup := .relation 113613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113613) (rhsResult := 113611)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 113612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩]⟩) (none) 113611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113614

namespace LeftMerge113615
def owner : Owner := ⟨.program ⟨257⟩, ⟨16302⟩⟩
def mergeEvent : Nat := 113615
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩] } }
def rhsRaw : List Term := Proof.Events443.exact113611RawTerms
def group : MergeGroup := .relation 113613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113613) (rhsResult := 113611)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 113612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩]⟩) (none) 113611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113615

namespace LeftMerge113616
def owner : Owner := ⟨.program ⟨257⟩, ⟨16302⟩⟩
def mergeEvent : Nat := 113616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16855⟩⟩] } }
def rhsRaw : List Term := Proof.Events443.exact113611RawTerms
def group : MergeGroup := .relation 113613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113613) (rhsResult := 113611)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 113612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩]⟩) (none) 113611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16855⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], [⟨.program ⟨257⟩, ⟨16855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113616

namespace LeftMerge113617
def owner : Owner := ⟨.program ⟨257⟩, ⟨16302⟩⟩
def mergeEvent : Nat := 113617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨15796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events443.exact113611RawTerms
def group : MergeGroup := .relation 113613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 113613) (rhsResult := 113611)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 113612 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16299⟩⟩]⟩) (none) 113611) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨15796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113617

namespace LeftMerge113622
def owner : Owner := ⟨.program ⟨257⟩, ⟨17372⟩⟩
def mergeEvent : Nat := 113622
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16855⟩⟩] } }
def leftRaw : List Term := Proof.Events443.exact113618RawTerms
def rightRaw : List Term := Proof.Events443.exact113432RawTerms
def group : MergeGroup := .operator 113618 113432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113618) (leftOrdinal := 2)
    (rightResult := 113432) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16855⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16855⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12396⟩⟩, ⟨.program ⟨257⟩, ⟨15498⟩⟩], [⟨.program ⟨257⟩, ⟨16855⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge113622

namespace LeftMerge113623
def owner : Owner := ⟨.program ⟨257⟩, ⟨17372⟩⟩
def mergeEvent : Nat := 113623
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩] } }
def leftRaw : List Term := Proof.Events443.exact113618RawTerms
def rightRaw : List Term := Proof.Events443.exact113432RawTerms
def group : MergeGroup := .operator 113618 113432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 113618) (leftOrdinal := 1)
    (rightResult := 113432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17370⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge113623

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
