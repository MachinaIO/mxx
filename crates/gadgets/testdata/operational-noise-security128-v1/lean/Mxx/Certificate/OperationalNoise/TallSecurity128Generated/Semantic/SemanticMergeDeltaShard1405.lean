import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge228502
def owner : Owner := ⟨.program ⟨257⟩, ⟨52509⟩⟩
def mergeEvent : Nat := 228502
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52003⟩⟩] } }
def rhsRaw : List Term := Proof.Events892.exact228427RawTerms
def group : MergeGroup := .relation 228501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 228501) (rhsResult := 228427)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52508⟩⟩) ⟨52003⟩ 228427) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge228502

namespace LeftMerge228503
def owner : Owner := ⟨.program ⟨257⟩, ⟨52509⟩⟩
def mergeEvent : Nat := 228503
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩] } }
def leftRaw : List Term := Proof.Events892.exact228494RawTerms
def rightRaw : List Term := Proof.Events892.exact228430RawTerms
def group : MergeGroup := .operator 228494 228430
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 228494) (leftOrdinal := 0)
    (rightResult := 228430) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52508⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228503

namespace LeftMerge228517
def owner : Owner := ⟨.program ⟨257⟩, ⟨51442⟩⟩
def mergeEvent : Nat := 228517
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events892.exact228511RawTerms
def group : MergeGroup := .operator 222245 228511
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 228511) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51439⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228517

namespace LeftMerge228596
def owner : Owner := ⟨.program ⟨257⟩, ⟨50519⟩⟩
def mergeEvent : Nat := 228596
def frameStart : Nat := 228566
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events892.exact228592RawTerms
def rightRaw : List Term := Proof.Events892.exact228589RawTerms
def group : MergeGroup := .operator 228592 228589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 228592) (leftOrdinal := 0)
    (rightResult := 228589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24518⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228596

namespace LeftMerge228626
def owner : Owner := ⟨.program ⟨257⟩, ⟨52284⟩⟩
def mergeEvent : Nat := 228626
def frameStart : Nat := 228566
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events893.exact228622RawTerms
def rightRaw : List Term := Proof.Events893.exact228620RawTerms
def group : MergeGroup := .operator 228622 228620
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 228622) (leftOrdinal := 0)
    (rightResult := 228620) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228626

namespace LeftMerge228649
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 228649
def frameStart : Nat := 228566
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events893.exact228645RawTerms
def rightRaw : List Term := Proof.Events893.exact228642RawTerms
def group : MergeGroup := .operator 228645 228642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 228645) (leftOrdinal := 0)
    (rightResult := 228642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228649

namespace LeftMerge228658
def owner : Owner := ⟨.program ⟨257⟩, ⟨52511⟩⟩
def mergeEvent : Nat := 228658
def frameStart : Nat := 228566
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩] } }
def leftRaw : List Term := Proof.Events893.exact228654RawTerms
def rightRaw : List Term := Proof.Events893.exact228611RawTerms
def group : MergeGroup := .operator 228654 228611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 228654) (leftOrdinal := 0)
    (rightResult := 228611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52508⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228658

namespace LeftMerge228659
def owner : Owner := ⟨.program ⟨257⟩, ⟨52511⟩⟩
def mergeEvent : Nat := 228659
def frameStart : Nat := 228566
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩] } }
def leftRaw : List Term := Proof.Events893.exact228654RawTerms
def rightRaw : List Term := Proof.Events893.exact228611RawTerms
def group : MergeGroup := .operator 228654 228611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 228654) (leftOrdinal := 1)
    (rightResult := 228611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52508⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge228659

namespace LeftMerge228661
def owner : Owner := ⟨.program ⟨257⟩, ⟨52511⟩⟩
def mergeEvent : Nat := 228661
def frameStart : Nat := 228566
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52003⟩⟩] } }
def rhsRaw : List Term := Proof.Events893.exact228608RawTerms
def group : MergeGroup := .relation 228660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 228660) (rhsResult := 228608)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52508⟩⟩) ⟨52003⟩ 228608) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge228661

namespace LeftMerge228669
def owner : Owner := ⟨.program ⟨257⟩, ⟨50882⟩⟩
def mergeEvent : Nat := 228669
def frameStart : Nat := 228566
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events893.exact228622RawTerms
def rightRaw : List Term := Proof.Events893.exact228665RawTerms
def group : MergeGroup := .operator 228622 228665
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 228622) (leftOrdinal := 0)
    (rightResult := 228665) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50880⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228669

namespace LeftMerge228686
def owner : Owner := ⟨.program ⟨257⟩, ⟨51442⟩⟩
def mergeEvent : Nat := 228686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events893.exact228683RawTerms
def group : MergeGroup := .relation 228685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 228685) (rhsResult := 228683)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 228684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩) (none) 228683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228686

namespace LeftMerge228687
def owner : Owner := ⟨.program ⟨257⟩, ⟨51442⟩⟩
def mergeEvent : Nat := 228687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩] } }
def rhsRaw : List Term := Proof.Events893.exact228683RawTerms
def group : MergeGroup := .relation 228685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 228685) (rhsResult := 228683)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 228684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩) (none) 228683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge228687

namespace LeftMerge228688
def owner : Owner := ⟨.program ⟨257⟩, ⟨51442⟩⟩
def mergeEvent : Nat := 228688
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52003⟩⟩] } }
def rhsRaw : List Term := Proof.Events893.exact228683RawTerms
def group : MergeGroup := .relation 228685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 228685) (rhsResult := 228683)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 228684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩) (none) 228683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228688

namespace LeftMerge228689
def owner : Owner := ⟨.program ⟨257⟩, ⟨51442⟩⟩
def mergeEvent : Nat := 228689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events893.exact228683RawTerms
def group : MergeGroup := .relation 228685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 228685) (rhsResult := 228683)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 228684 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51439⟩⟩]⟩) (none) 228683) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨50880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge228689

namespace LeftMerge228694
def owner : Owner := ⟨.program ⟨257⟩, ⟨52510⟩⟩
def mergeEvent : Nat := 228694
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52003⟩⟩] } }
def leftRaw : List Term := Proof.Events893.exact228690RawTerms
def rightRaw : List Term := Proof.Events892.exact228504RawTerms
def group : MergeGroup := .operator 228690 228504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 228690) (leftOrdinal := 2)
    (rightResult := 228504) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52003⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52003⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨24518⟩⟩, ⟨.program ⟨257⟩, ⟨50518⟩⟩], [⟨.program ⟨257⟩, ⟨52003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge228694

namespace LeftMerge228695
def owner : Owner := ⟨.program ⟨257⟩, ⟨52510⟩⟩
def mergeEvent : Nat := 228695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩] } }
def leftRaw : List Term := Proof.Events893.exact228690RawTerms
def rightRaw : List Term := Proof.Events892.exact228504RawTerms
def group : MergeGroup := .operator 228690 228504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 228690) (leftOrdinal := 1)
    (rightResult := 228504) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52508⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge228695

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
