import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge127760
def owner : Owner := ⟨.program ⟨257⟩, ⟨19112⟩⟩
def mergeEvent : Nat := 127760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127754RawTerms
def group : MergeGroup := .relation 127756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127756) (rhsResult := 127754)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127755 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19109⟩⟩]⟩) (none) 127754) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127760

namespace LeftMerge127765
def owner : Owner := ⟨.program ⟨257⟩, ⟨20177⟩⟩
def mergeEvent : Nat := 127765
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19685⟩⟩] } }
def leftRaw : List Term := Proof.Events499.exact127761RawTerms
def rightRaw : List Term := Proof.Events498.exact127575RawTerms
def group : MergeGroup := .operator 127761 127575
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127761) (leftOrdinal := 2)
    (rightResult := 127575) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19685⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19685⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨12621⟩⟩, ⟨.program ⟨257⟩, ⟨18178⟩⟩], [⟨.program ⟨257⟩, ⟨19685⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127765

namespace LeftMerge127766
def owner : Owner := ⟨.program ⟨257⟩, ⟨20177⟩⟩
def mergeEvent : Nat := 127766
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩] } }
def leftRaw : List Term := Proof.Events499.exact127761RawTerms
def rightRaw : List Term := Proof.Events498.exact127575RawTerms
def group : MergeGroup := .operator 127761 127575
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127761) (leftOrdinal := 1)
    (rightResult := 127575) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20175⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127766

namespace LeftMerge127774
def owner : Owner := ⟨.program ⟨257⟩, ⟨20530⟩⟩
def mergeEvent : Nat := 127774
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩] } }
def leftRaw : List Term := Proof.Events499.exact127768RawTerms
def rightRaw : List Term := Proof.Events498.exact127491RawTerms
def group : MergeGroup := .operator 127768 127491
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127768) (leftOrdinal := 0)
    (rightResult := 127491) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20528⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127774

namespace LeftMerge127775
def owner : Owner := ⟨.program ⟨257⟩, ⟨20530⟩⟩
def mergeEvent : Nat := 127775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩] } }
def leftRaw : List Term := Proof.Events499.exact127768RawTerms
def rightRaw : List Term := Proof.Events498.exact127491RawTerms
def group : MergeGroup := .operator 127768 127491
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127768) (leftOrdinal := 1)
    (rightResult := 127491) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20528⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127775

namespace LeftMerge127777
def owner : Owner := ⟨.program ⟨257⟩, ⟨20530⟩⟩
def mergeEvent : Nat := 127777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19825⟩⟩] } }
def rhsRaw : List Term := Proof.Events498.exact127488RawTerms
def group : MergeGroup := .relation 127776
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127776) (rhsResult := 127488)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20528⟩⟩) ⟨19825⟩ 127488) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19825⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨19825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127777

namespace LeftMerge127791
def owner : Owner := ⟨.program ⟨257⟩, ⟨19379⟩⟩
def mergeEvent : Nat := 127791
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events499.exact127785RawTerms
def group : MergeGroup := .operator 119870 127785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 127785) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19376⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127791

namespace LeftMerge127912
def owner : Owner := ⟨.program ⟨257⟩, ⟨20052⟩⟩
def mergeEvent : Nat := 127912
def frameStart : Nat := 127846
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events499.exact127908RawTerms
def rightRaw : List Term := Proof.Events499.exact127906RawTerms
def group : MergeGroup := .operator 127908 127906
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127908) (leftOrdinal := 0)
    (rightResult := 127906) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127912

namespace LeftMerge127924
def owner : Owner := ⟨.program ⟨257⟩, ⟨20529⟩⟩
def mergeEvent : Nat := 127924
def frameStart : Nat := 127846
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩] } }
def leftRaw : List Term := Proof.Events499.exact127920RawTerms
def rightRaw : List Term := Proof.Events499.exact127897RawTerms
def group : MergeGroup := .operator 127920 127897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127920) (leftOrdinal := 0)
    (rightResult := 127897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20528⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127924

namespace LeftMerge127925
def owner : Owner := ⟨.program ⟨257⟩, ⟨20529⟩⟩
def mergeEvent : Nat := 127925
def frameStart : Nat := 127846
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩] } }
def leftRaw : List Term := Proof.Events499.exact127920RawTerms
def rightRaw : List Term := Proof.Events499.exact127897RawTerms
def group : MergeGroup := .operator 127920 127897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127920) (leftOrdinal := 1)
    (rightResult := 127897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20528⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127925

namespace LeftMerge127927
def owner : Owner := ⟨.program ⟨257⟩, ⟨20529⟩⟩
def mergeEvent : Nat := 127927
def frameStart : Nat := 127846
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19825⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127894RawTerms
def group : MergeGroup := .relation 127926
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127926) (rhsResult := 127894)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20528⟩⟩) ⟨19825⟩ 127894) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19825⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨19825⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127927

namespace LeftMerge127935
def owner : Owner := ⟨.program ⟨257⟩, ⟨18792⟩⟩
def mergeEvent : Nat := 127935
def frameStart : Nat := 127846
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18790⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events499.exact127908RawTerms
def rightRaw : List Term := Proof.Events499.exact127931RawTerms
def group : MergeGroup := .operator 127908 127931
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 127908) (leftOrdinal := 0)
    (rightResult := 127931) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18790⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127935

namespace LeftMerge127952
def owner : Owner := ⟨.program ⟨257⟩, ⟨19379⟩⟩
def mergeEvent : Nat := 127952
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127949RawTerms
def group : MergeGroup := .relation 127951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127951) (rhsResult := 127949)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127950 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩]⟩) (none) 127949) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127952

namespace LeftMerge127953
def owner : Owner := ⟨.program ⟨257⟩, ⟨19379⟩⟩
def mergeEvent : Nat := 127953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127949RawTerms
def group : MergeGroup := .relation 127951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127951) (rhsResult := 127949)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127950 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩]⟩) (none) 127949) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20528⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127953

namespace LeftMerge127954
def owner : Owner := ⟨.program ⟨257⟩, ⟨19379⟩⟩
def mergeEvent : Nat := 127954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19825⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127949RawTerms
def group : MergeGroup := .relation 127951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127951) (rhsResult := 127949)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127950 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩]⟩) (none) 127949) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18556⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19825⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18556⟩⟩], [⟨.program ⟨257⟩, ⟨19825⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge127954

namespace LeftMerge127955
def owner : Owner := ⟨.program ⟨257⟩, ⟨19379⟩⟩
def mergeEvent : Nat := 127955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events499.exact127949RawTerms
def group : MergeGroup := .relation 127951
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 127951) (rhsResult := 127949)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 127950 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19376⟩⟩]⟩) (none) 127949) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18790⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨18790⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge127955

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
