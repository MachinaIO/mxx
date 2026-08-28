import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge8480
def owner : Owner := ⟨.program ⟨214⟩, ⟨7394⟩⟩
def mergeEvent : Nat := 8480
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events033.exact8476RawTerms
def group : MergeGroup := .operator 6314 8476
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 8476) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6786⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8480

namespace LeftMerge8497
def owner : Owner := ⟨.program ⟨214⟩, ⟨12604⟩⟩
def mergeEvent : Nat := 8497
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8491RawTerms
def rightRaw : List Term := Proof.Events000.exact146RawTerms
def group : MergeGroup := .operator 8491 146
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8491) (leftOrdinal := 1)
    (rightResult := 146) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8497

namespace LeftMerge8498
def owner : Owner := ⟨.program ⟨214⟩, ⟨12604⟩⟩
def mergeEvent : Nat := 8498
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8491RawTerms
def rightRaw : List Term := Proof.Events000.exact146RawTerms
def group : MergeGroup := .operator 8491 146
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8491) (leftOrdinal := 0)
    (rightResult := 146) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], [⟨.program ⟨214⟩, ⟨6786⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8498

namespace LeftMerge8513
def owner : Owner := ⟨.program ⟨214⟩, ⟨9946⟩⟩
def mergeEvent : Nat := 8513
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact146RawTerms
def rightRaw : List Term := Proof.Events025.exact6449RawTerms
def group : MergeGroup := .operator 146 6449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 146) (leftOrdinal := 0)
    (rightResult := 6449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8513

namespace LeftMerge8521
def owner : Owner := ⟨.program ⟨214⟩, ⟨7374⟩⟩
def mergeEvent : Nat := 8521
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events033.exact8517RawTerms
def group : MergeGroup := .operator 6314 8517
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 8517) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8521

namespace LeftMerge8538
def owner : Owner := ⟨.program ⟨214⟩, ⟨9949⟩⟩
def mergeEvent : Nat := 8538
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8532RawTerms
def rightRaw : List Term := Proof.Events033.exact8506RawTerms
def group : MergeGroup := .operator 8532 8506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8532) (leftOrdinal := 1)
    (rightResult := 8506) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7870⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8538

namespace LeftMerge8540
def owner : Owner := ⟨.program ⟨214⟩, ⟨9949⟩⟩
def mergeEvent : Nat := 8540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }
def rhsRaw : List Term := Proof.Events033.exact8476RawTerms
def group : MergeGroup := .relation 8539
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 8539) (rhsResult := 8476)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7870⟩⟩) ⟨6786⟩ 8476) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], [⟨.program ⟨214⟩, ⟨6786⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8540

namespace LeftMerge8541
def owner : Owner := ⟨.program ⟨214⟩, ⟨9949⟩⟩
def mergeEvent : Nat := 8541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8532RawTerms
def rightRaw : List Term := Proof.Events033.exact8506RawTerms
def group : MergeGroup := .operator 8532 8506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8532) (leftOrdinal := 0)
    (rightResult := 8506) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7870⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8541

namespace LeftMerge8546
def owner : Owner := ⟨.program ⟨214⟩, ⟨12605⟩⟩
def mergeEvent : Nat := 8546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8542RawTerms
def rightRaw : List Term := Proof.Events033.exact8499RawTerms
def group : MergeGroup := .operator 8542 8499
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8542) (leftOrdinal := 1)
    (rightResult := 8499) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6786⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩], [⟨.program ⟨214⟩, ⟨6786⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8546

namespace LeftMerge8554
def owner : Owner := ⟨.program ⟨214⟩, ⟨25471⟩⟩
def mergeEvent : Nat := 8554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8548RawTerms
def rightRaw : List Term := Proof.Events033.exact8465RawTerms
def group : MergeGroup := .operator 8548 8465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8548) (leftOrdinal := 1)
    (rightResult := 8465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25470⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8554

namespace LeftMerge8556
def owner : Owner := ⟨.program ⟨214⟩, ⟨25471⟩⟩
def mergeEvent : Nat := 8556
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23256⟩⟩] } }
def rhsRaw : List Term := Proof.Events033.exact8462RawTerms
def group : MergeGroup := .relation 8555
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 8555) (rhsResult := 8462)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25470⟩⟩) ⟨23256⟩ 8462) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23256⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨23256⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge8556

namespace LeftMerge8557
def owner : Owner := ⟨.program ⟨214⟩, ⟨25471⟩⟩
def mergeEvent : Nat := 8557
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8548RawTerms
def rightRaw : List Term := Proof.Events033.exact8465RawTerms
def group : MergeGroup := .operator 8548 8465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8548) (leftOrdinal := 0)
    (rightResult := 8465) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25470⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩, ⟨.program ⟨214⟩, ⟨25470⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8557

namespace LeftMerge8571
def owner : Owner := ⟨.program ⟨214⟩, ⟨19979⟩⟩
def mergeEvent : Nat := 8571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events033.exact8565RawTerms
def group : MergeGroup := .operator 6561 8565
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 8565) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19976⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19976⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8571

namespace LeftMerge8650
def owner : Owner := ⟨.program ⟨214⟩, ⟨12599⟩⟩
def mergeEvent : Nat := 8650
def frameStart : Nat := 8620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events033.exact8646RawTerms
def rightRaw : List Term := Proof.Events033.exact8643RawTerms
def group : MergeGroup := .operator 8646 8643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8646) (leftOrdinal := 0)
    (rightResult := 8643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8650

namespace LeftMerge8680
def owner : Owner := ⟨.program ⟨214⟩, ⟨12680⟩⟩
def mergeEvent : Nat := 8680
def frameStart : Nat := 8620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8676RawTerms
def rightRaw : List Term := Proof.Events033.exact8674RawTerms
def group : MergeGroup := .operator 8676 8674
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8676) (leftOrdinal := 0)
    (rightResult := 8674) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9945⟩⟩, ⟨.program ⟨214⟩, ⟨12598⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8680

namespace LeftMerge8703
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def mergeEvent : Nat := 8703
def frameStart : Nat := 8620
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8699RawTerms
def rightRaw : List Term := Proof.Events033.exact8696RawTerms
def group : MergeGroup := .operator 8699 8696
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8699) (leftOrdinal := 0)
    (rightResult := 8696) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6766⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7870⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6766⟩⟩, ⟨.program ⟨214⟩, ⟨7870⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge8703

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
