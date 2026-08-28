import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge120561
def owner : Owner := ⟨.program ⟨257⟩, ⟨46139⟩⟩
def mergeEvent : Nat := 120561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩] } }
def leftRaw : List Term := Proof.Events468.exact119870RawTerms
def rightRaw : List Term := Proof.Events470.exact120555RawTerms
def group : MergeGroup := .operator 119870 120555
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119870) (leftOrdinal := 0)
    (rightResult := 120555) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46136⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120561

namespace LeftMerge120682
def owner : Owner := ⟨.program ⟨257⟩, ⟨46812⟩⟩
def mergeEvent : Nat := 120682
def frameStart : Nat := 120616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events471.exact120678RawTerms
def rightRaw : List Term := Proof.Events471.exact120676RawTerms
def group : MergeGroup := .operator 120678 120676
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120678) (leftOrdinal := 0)
    (rightResult := 120676) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120682

namespace LeftMerge120694
def owner : Owner := ⟨.program ⟨257⟩, ⟨47250⟩⟩
def mergeEvent : Nat := 120694
def frameStart : Nat := 120616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩] } }
def leftRaw : List Term := Proof.Events471.exact120690RawTerms
def rightRaw : List Term := Proof.Events471.exact120667RawTerms
def group : MergeGroup := .operator 120690 120667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120690) (leftOrdinal := 0)
    (rightResult := 120667) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47249⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120694

namespace LeftMerge120695
def owner : Owner := ⟨.program ⟨257⟩, ⟨47250⟩⟩
def mergeEvent : Nat := 120695
def frameStart : Nat := 120616
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩] } }
def leftRaw : List Term := Proof.Events471.exact120690RawTerms
def rightRaw : List Term := Proof.Events471.exact120667RawTerms
def group : MergeGroup := .operator 120690 120667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120690) (leftOrdinal := 1)
    (rightResult := 120667) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120695

namespace LeftMerge120697
def owner : Owner := ⟨.program ⟨257⟩, ⟨47250⟩⟩
def mergeEvent : Nat := 120697
def frameStart : Nat := 120616
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46585⟩⟩] } }
def rhsRaw : List Term := Proof.Events471.exact120664RawTerms
def group : MergeGroup := .relation 120696
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120696) (rhsResult := 120664)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47249⟩⟩) ⟨46585⟩ 120664) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46585⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨46585⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120697

namespace LeftMerge120705
def owner : Owner := ⟨.program ⟨257⟩, ⟨45632⟩⟩
def mergeEvent : Nat := 120705
def frameStart : Nat := 120616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45631⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events471.exact120678RawTerms
def rightRaw : List Term := Proof.Events471.exact120701RawTerms
def group : MergeGroup := .operator 120678 120701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120678) (leftOrdinal := 0)
    (rightResult := 120701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45631⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45631⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120705

namespace LeftMerge120722
def owner : Owner := ⟨.program ⟨257⟩, ⟨46139⟩⟩
def mergeEvent : Nat := 120722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }
def rhsRaw : List Term := Proof.Events471.exact120719RawTerms
def group : MergeGroup := .relation 120721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120721) (rhsResult := 120719)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120720 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩]⟩) (none) 120719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120722

namespace LeftMerge120723
def owner : Owner := ⟨.program ⟨257⟩, ⟨46139⟩⟩
def mergeEvent : Nat := 120723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩] } }
def rhsRaw : List Term := Proof.Events471.exact120719RawTerms
def group : MergeGroup := .relation 120721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120721) (rhsResult := 120719)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120720 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩]⟩) (none) 120719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120723

namespace LeftMerge120724
def owner : Owner := ⟨.program ⟨257⟩, ⟨46139⟩⟩
def mergeEvent : Nat := 120724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46585⟩⟩] } }
def rhsRaw : List Term := Proof.Events471.exact120719RawTerms
def group : MergeGroup := .relation 120721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120721) (rhsResult := 120719)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120720 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩]⟩) (none) 120719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46585⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨46585⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120724

namespace LeftMerge120725
def owner : Owner := ⟨.program ⟨257⟩, ⟨46139⟩⟩
def mergeEvent : Nat := 120725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45631⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events471.exact120719RawTerms
def group : MergeGroup := .relation 120721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 120721) (rhsResult := 120719)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 120720 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46136⟩⟩]⟩) (none) 120719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45631⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45631⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120725

namespace LeftMerge120730
def owner : Owner := ⟨.program ⟨257⟩, ⟨47252⟩⟩
def mergeEvent : Nat := 120730
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩] } }
def leftRaw : List Term := Proof.Events471.exact120726RawTerms
def rightRaw : List Term := Proof.Events470.exact120548RawTerms
def group : MergeGroup := .operator 120726 120548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120726) (leftOrdinal := 0)
    (rightResult := 120548) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47249⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120730

namespace LeftMerge120731
def owner : Owner := ⟨.program ⟨257⟩, ⟨47252⟩⟩
def mergeEvent : Nat := 120731
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46585⟩⟩] } }
def leftRaw : List Term := Proof.Events471.exact120726RawTerms
def rightRaw : List Term := Proof.Events470.exact120548RawTerms
def group : MergeGroup := .operator 120726 120548
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120726) (leftOrdinal := 2)
    (rightResult := 120548) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46585⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46585⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨45436⟩⟩], [⟨.program ⟨257⟩, ⟨46585⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120731

namespace LeftMerge120757
def owner : Owner := ⟨.program ⟨257⟩, ⟨42381⟩⟩
def mergeEvent : Nat := 120757
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events021.exact5376RawTerms
def rightRaw : List Term := Proof.Events467.exact119778RawTerms
def group : MergeGroup := .operator 5376 119778
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 5376) (leftOrdinal := 0)
    (rightResult := 119778) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42378⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120757

namespace LeftMerge120762
def owner : Owner := ⟨.program ⟨257⟩, ⟨8133⟩⟩
def mergeEvent : Nat := 120762
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events467.exact119648RawTerms
def rightRaw : List Term := Proof.Events070.exact18082RawTerms
def group : MergeGroup := .operator 119648 18082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 119648) (leftOrdinal := 0)
    (rightResult := 18082) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120762

namespace LeftMerge120779
def owner : Owner := ⟨.program ⟨257⟩, ⟨42384⟩⟩
def mergeEvent : Nat := 120779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events471.exact120773RawTerms
def rightRaw : List Term := Proof.Events021.exact5379RawTerms
def group : MergeGroup := .operator 120773 5379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120773) (leftOrdinal := 1)
    (rightResult := 5379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14421⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩, ⟨.program ⟨257⟩, ⟨42378⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge120779

namespace LeftMerge120780
def owner : Owner := ⟨.program ⟨257⟩, ⟨42384⟩⟩
def mergeEvent : Nat := 120780
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events471.exact120773RawTerms
def rightRaw : List Term := Proof.Events021.exact5379RawTerms
def group : MergeGroup := .operator 120773 5379
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 120773) (leftOrdinal := 0)
    (rightResult := 5379) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14421⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5757⟩⟩, ⟨.program ⟨257⟩, ⟨14421⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge120780

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
