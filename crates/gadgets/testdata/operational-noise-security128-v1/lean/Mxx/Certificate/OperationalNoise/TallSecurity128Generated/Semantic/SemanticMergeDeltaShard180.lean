import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge32594
def owner : Owner := ⟨.program ⟨257⟩, ⟨47079⟩⟩
def mergeEvent : Nat := 32594
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩] } }
def leftRaw : List Term := Proof.Events127.exact32585RawTerms
def rightRaw : List Term := Proof.Events127.exact32521RawTerms
def group : MergeGroup := .operator 32585 32521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32585) (leftOrdinal := 0)
    (rightResult := 32521) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47078⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32594

namespace LeftMerge32608
def owner : Owner := ⟨.program ⟨257⟩, ⟨46002⟩⟩
def mergeEvent : Nat := 32608
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events127.exact32602RawTerms
def group : MergeGroup := .operator 32120 32602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 32602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45999⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32608

namespace LeftMerge32687
def owner : Owner := ⟨.program ⟨257⟩, ⟨45371⟩⟩
def mergeEvent : Nat := 32687
def frameStart : Nat := 32657
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events127.exact32683RawTerms
def rightRaw : List Term := Proof.Events127.exact32680RawTerms
def group : MergeGroup := .operator 32683 32680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32683) (leftOrdinal := 0)
    (rightResult := 32680) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14916⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32687

namespace LeftMerge32717
def owner : Owner := ⟨.program ⟨257⟩, ⟨46784⟩⟩
def mergeEvent : Nat := 32717
def frameStart : Nat := 32657
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events127.exact32713RawTerms
def rightRaw : List Term := Proof.Events127.exact32711RawTerms
def group : MergeGroup := .operator 32713 32711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32713) (leftOrdinal := 0)
    (rightResult := 32711) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32717

namespace LeftMerge32740
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 32740
def frameStart : Nat := 32657
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events127.exact32736RawTerms
def rightRaw : List Term := Proof.Events127.exact32733RawTerms
def group : MergeGroup := .operator 32736 32733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32736) (leftOrdinal := 0)
    (rightResult := 32733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32740

namespace LeftMerge32749
def owner : Owner := ⟨.program ⟨257⟩, ⟨47081⟩⟩
def mergeEvent : Nat := 32749
def frameStart : Nat := 32657
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩] } }
def leftRaw : List Term := Proof.Events127.exact32745RawTerms
def rightRaw : List Term := Proof.Events127.exact32702RawTerms
def group : MergeGroup := .operator 32745 32702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32745) (leftOrdinal := 0)
    (rightResult := 32702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47078⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32749

namespace LeftMerge32750
def owner : Owner := ⟨.program ⟨257⟩, ⟨47081⟩⟩
def mergeEvent : Nat := 32750
def frameStart : Nat := 32657
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩] } }
def leftRaw : List Term := Proof.Events127.exact32745RawTerms
def rightRaw : List Term := Proof.Events127.exact32702RawTerms
def group : MergeGroup := .operator 32745 32702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32745) (leftOrdinal := 1)
    (rightResult := 32702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47078⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32750

namespace LeftMerge32752
def owner : Owner := ⟨.program ⟨257⟩, ⟨47081⟩⟩
def mergeEvent : Nat := 32752
def frameStart : Nat := 32657
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46523⟩⟩] } }
def rhsRaw : List Term := Proof.Events127.exact32699RawTerms
def group : MergeGroup := .relation 32751
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 32751) (rhsResult := 32699)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47078⟩⟩) ⟨46523⟩ 32699) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46523⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], [⟨.program ⟨257⟩, ⟨46523⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32752

namespace LeftMerge32760
def owner : Owner := ⟨.program ⟨257⟩, ⟨45542⟩⟩
def mergeEvent : Nat := 32760
def frameStart : Nat := 32657
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events127.exact32713RawTerms
def rightRaw : List Term := Proof.Events127.exact32756RawTerms
def group : MergeGroup := .operator 32713 32756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32713) (leftOrdinal := 0)
    (rightResult := 32756) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45540⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45540⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32760

namespace LeftMerge32777
def owner : Owner := ⟨.program ⟨257⟩, ⟨46002⟩⟩
def mergeEvent : Nat := 32777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events128.exact32774RawTerms
def group : MergeGroup := .relation 32776
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 32776) (rhsResult := 32774)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 32775 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩]⟩) (none) 32774) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32777

namespace LeftMerge32778
def owner : Owner := ⟨.program ⟨257⟩, ⟨46002⟩⟩
def mergeEvent : Nat := 32778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩] } }
def rhsRaw : List Term := Proof.Events128.exact32774RawTerms
def group : MergeGroup := .relation 32776
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 32776) (rhsResult := 32774)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 32775 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩]⟩) (none) 32774) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32778

namespace LeftMerge32779
def owner : Owner := ⟨.program ⟨257⟩, ⟨46002⟩⟩
def mergeEvent : Nat := 32779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46523⟩⟩] } }
def rhsRaw : List Term := Proof.Events128.exact32774RawTerms
def group : MergeGroup := .relation 32776
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 32776) (rhsResult := 32774)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 32775 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩]⟩) (none) 32774) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46523⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], [⟨.program ⟨257⟩, ⟨46523⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32779

namespace LeftMerge32780
def owner : Owner := ⟨.program ⟨257⟩, ⟨46002⟩⟩
def mergeEvent : Nat := 32780
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events128.exact32774RawTerms
def group : MergeGroup := .relation 32776
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 32776) (rhsResult := 32774)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 32775 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45999⟩⟩]⟩) (none) 32774) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45540⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨45540⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32780

namespace LeftMerge32785
def owner : Owner := ⟨.program ⟨257⟩, ⟨47080⟩⟩
def mergeEvent : Nat := 32785
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46523⟩⟩] } }
def leftRaw : List Term := Proof.Events128.exact32781RawTerms
def rightRaw : List Term := Proof.Events127.exact32595RawTerms
def group : MergeGroup := .operator 32781 32595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32781) (leftOrdinal := 2)
    (rightResult := 32595) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46523⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46523⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14916⟩⟩, ⟨.program ⟨257⟩, ⟨45370⟩⟩], [⟨.program ⟨257⟩, ⟨46523⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge32785

namespace LeftMerge32786
def owner : Owner := ⟨.program ⟨257⟩, ⟨47080⟩⟩
def mergeEvent : Nat := 32786
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩] } }
def leftRaw : List Term := Proof.Events128.exact32781RawTerms
def rightRaw : List Term := Proof.Events127.exact32595RawTerms
def group : MergeGroup := .operator 32781 32595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32781) (leftOrdinal := 1)
    (rightResult := 32595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47078⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32786

namespace LeftMerge32794
def owner : Owner := ⟨.program ⟨257⟩, ⟨47576⟩⟩
def mergeEvent : Nat := 32794
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47574⟩⟩] } }
def leftRaw : List Term := Proof.Events128.exact32788RawTerms
def rightRaw : List Term := Proof.Events126.exact32511RawTerms
def group : MergeGroup := .operator 32788 32511
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32788) (leftOrdinal := 0)
    (rightResult := 32511) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47574⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge32794

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
