import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge98480
def owner : Owner := ⟨.program ⟨257⟩, ⟨20277⟩⟩
def mergeEvent : Nat := 98480
def frameStart : Nat := 98387
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98475RawTerms
def rightRaw : List Term := Proof.Events384.exact98432RawTerms
def group : MergeGroup := .operator 98475 98432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98475) (leftOrdinal := 1)
    (rightResult := 98432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98480

namespace LeftMerge98482
def owner : Owner := ⟨.program ⟨257⟩, ⟨20277⟩⟩
def mergeEvent : Nat := 98482
def frameStart : Nat := 98387
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19739⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98429RawTerms
def group : MergeGroup := .relation 98481
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98481) (rhsResult := 98429)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20274⟩⟩) ⟨19739⟩ 98429) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19739⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨19739⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98482

namespace LeftMerge98490
def owner : Owner := ⟨.program ⟨257⟩, ⟨18630⟩⟩
def mergeEvent : Nat := 98490
def frameStart : Nat := 98387
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98443RawTerms
def rightRaw : List Term := Proof.Events384.exact98486RawTerms
def group : MergeGroup := .operator 98443 98486
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98443) (leftOrdinal := 0)
    (rightResult := 98486) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18628⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98490

namespace LeftMerge98507
def owner : Owner := ⟨.program ⟨257⟩, ⟨19202⟩⟩
def mergeEvent : Nat := 98507
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98504RawTerms
def group : MergeGroup := .relation 98506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98506) (rhsResult := 98504)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19199⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19199⟩⟩]⟩) (none) 98504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98507

namespace LeftMerge98508
def owner : Owner := ⟨.program ⟨257⟩, ⟨19202⟩⟩
def mergeEvent : Nat := 98508
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98504RawTerms
def group : MergeGroup := .relation 98506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98506) (rhsResult := 98504)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19199⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19199⟩⟩]⟩) (none) 98504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98508

namespace LeftMerge98509
def owner : Owner := ⟨.program ⟨257⟩, ⟨19202⟩⟩
def mergeEvent : Nat := 98509
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19739⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98504RawTerms
def group : MergeGroup := .relation 98506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98506) (rhsResult := 98504)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19199⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19199⟩⟩]⟩) (none) 98504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19739⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨19739⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98509

namespace LeftMerge98510
def owner : Owner := ⟨.program ⟨257⟩, ⟨19202⟩⟩
def mergeEvent : Nat := 98510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events384.exact98504RawTerms
def group : MergeGroup := .relation 98506
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98506) (rhsResult := 98504)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19199⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98505 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19199⟩⟩]⟩) (none) 98504) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18628⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98510

namespace LeftMerge98515
def owner : Owner := ⟨.program ⟨257⟩, ⟨20276⟩⟩
def mergeEvent : Nat := 98515
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19739⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98511RawTerms
def rightRaw : List Term := Proof.Events384.exact98325RawTerms
def group : MergeGroup := .operator 98511 98325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98511) (leftOrdinal := 2)
    (rightResult := 98325) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19739⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19739⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12756⟩⟩, ⟨.program ⟨257⟩, ⟨18394⟩⟩], [⟨.program ⟨257⟩, ⟨19739⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98515

namespace LeftMerge98516
def owner : Owner := ⟨.program ⟨257⟩, ⟨20276⟩⟩
def mergeEvent : Nat := 98516
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98511RawTerms
def rightRaw : List Term := Proof.Events384.exact98325RawTerms
def group : MergeGroup := .operator 98511 98325
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98511) (leftOrdinal := 1)
    (rightResult := 98325) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98516

namespace LeftMerge98524
def owner : Owner := ⟨.program ⟨257⟩, ⟨20809⟩⟩
def mergeEvent : Nat := 98524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20807⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98518RawTerms
def rightRaw : List Term := Proof.Events383.exact98241RawTerms
def group : MergeGroup := .operator 98518 98241
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98518) (leftOrdinal := 0)
    (rightResult := 98241) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20807⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20807⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98524

namespace LeftMerge98525
def owner : Owner := ⟨.program ⟨257⟩, ⟨20809⟩⟩
def mergeEvent : Nat := 98525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20807⟩⟩] } }
def leftRaw : List Term := Proof.Events384.exact98518RawTerms
def rightRaw : List Term := Proof.Events383.exact98241RawTerms
def group : MergeGroup := .operator 98518 98241
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98518) (leftOrdinal := 1)
    (rightResult := 98241) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20807⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18628⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20807⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98525

namespace LeftMerge98527
def owner : Owner := ⟨.program ⟨257⟩, ⟨20809⟩⟩
def mergeEvent : Nat := 98527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19906⟩⟩] } }
def rhsRaw : List Term := Proof.Events383.exact98238RawTerms
def group : MergeGroup := .relation 98526
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98526) (rhsResult := 98238)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18628⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20807⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20807⟩⟩) ⟨19906⟩ 98238) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19906⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18628⟩⟩], [⟨.program ⟨257⟩, ⟨19906⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98527

namespace LeftMerge98541
def owner : Owner := ⟨.program ⟨257⟩, ⟨19559⟩⟩
def mergeEvent : Nat := 98541
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19556⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events384.exact98535RawTerms
def group : MergeGroup := .operator 90620 98535
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 98535) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98541

namespace LeftMerge98662
def owner : Owner := ⟨.program ⟨257⟩, ⟨20088⟩⟩
def mergeEvent : Nat := 98662
def frameStart : Nat := 98596
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98658RawTerms
def rightRaw : List Term := Proof.Events385.exact98656RawTerms
def group : MergeGroup := .operator 98658 98656
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98658) (leftOrdinal := 0)
    (rightResult := 98656) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18628⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98662

namespace LeftMerge98674
def owner : Owner := ⟨.program ⟨257⟩, ⟨20808⟩⟩
def mergeEvent : Nat := 98674
def frameStart : Nat := 98596
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20807⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98670RawTerms
def rightRaw : List Term := Proof.Events385.exact98647RawTerms
def group : MergeGroup := .operator 98670 98647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98670) (leftOrdinal := 0)
    (rightResult := 98647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20807⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20807⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98674

namespace LeftMerge98675
def owner : Owner := ⟨.program ⟨257⟩, ⟨20808⟩⟩
def mergeEvent : Nat := 98675
def frameStart : Nat := 98596
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20807⟩⟩] } }
def leftRaw : List Term := Proof.Events385.exact98670RawTerms
def rightRaw : List Term := Proof.Events385.exact98647RawTerms
def group : MergeGroup := .operator 98670 98647
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98670) (leftOrdinal := 1)
    (rightResult := 98647) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18628⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20807⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18628⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20807⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98675

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
