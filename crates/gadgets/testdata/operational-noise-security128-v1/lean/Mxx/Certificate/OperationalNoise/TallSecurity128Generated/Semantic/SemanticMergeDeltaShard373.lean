import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge63668
def owner : Owner := ⟨.program ⟨257⟩, ⟨38319⟩⟩
def mergeEvent : Nat := 63668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }
def rhsRaw : List Term := Proof.Events248.exact63665RawTerms
def group : MergeGroup := .relation 63667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63667) (rhsResult := 63665)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63666 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38316⟩⟩]⟩) (none) 63665) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63668

namespace LeftMerge63669
def owner : Owner := ⟨.program ⟨257⟩, ⟨38319⟩⟩
def mergeEvent : Nat := 63669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39484⟩⟩] } }
def rhsRaw : List Term := Proof.Events248.exact63665RawTerms
def group : MergeGroup := .relation 63667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63667) (rhsResult := 63665)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63666 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38316⟩⟩]⟩) (none) 63665) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39484⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39484⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63669

namespace LeftMerge63670
def owner : Owner := ⟨.program ⟨257⟩, ⟨38319⟩⟩
def mergeEvent : Nat := 63670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38644⟩⟩] } }
def rhsRaw : List Term := Proof.Events248.exact63665RawTerms
def group : MergeGroup := .relation 63667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63667) (rhsResult := 63665)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63666 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38316⟩⟩]⟩) (none) 63665) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38644⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37484⟩⟩], [⟨.program ⟨257⟩, ⟨38644⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63670

namespace LeftMerge63671
def owner : Owner := ⟨.program ⟨257⟩, ⟨38319⟩⟩
def mergeEvent : Nat := 63671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events248.exact63665RawTerms
def group : MergeGroup := .relation 63667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63667) (rhsResult := 63665)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38316⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 63666 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38316⟩⟩]⟩) (none) 63665) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37734⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37734⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63671

namespace LeftMerge63676
def owner : Owner := ⟨.program ⟨257⟩, ⟨39487⟩⟩
def mergeEvent : Nat := 63676
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39484⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63672RawTerms
def rightRaw : List Term := Proof.Events248.exact63494RawTerms
def group : MergeGroup := .operator 63672 63494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63672) (leftOrdinal := 0)
    (rightResult := 63494) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39484⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39484⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39484⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63676

namespace LeftMerge63677
def owner : Owner := ⟨.program ⟨257⟩, ⟨39487⟩⟩
def mergeEvent : Nat := 63677
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38644⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63672RawTerms
def rightRaw : List Term := Proof.Events248.exact63494RawTerms
def group : MergeGroup := .operator 63672 63494
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63672) (leftOrdinal := 2)
    (rightResult := 63494) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38644⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38644⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨37484⟩⟩], [⟨.program ⟨257⟩, ⟨38644⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63677

namespace LeftMerge63703
def owner : Owner := ⟨.program ⟨257⟩, ⟨34605⟩⟩
def mergeEvent : Nat := 63703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2453RawTerms
def rightRaw : List Term := Proof.Events239.exact61278RawTerms
def group : MergeGroup := .operator 2453 61278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2453) (leftOrdinal := 0)
    (rightResult := 61278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63703

namespace LeftMerge63708
def owner : Owner := ⟨.program ⟨257⟩, ⟨10762⟩⟩
def mergeEvent : Nat := 63708
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61148RawTerms
def rightRaw : List Term := Proof.Events076.exact19585RawTerms
def group : MergeGroup := .operator 61148 19585
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61148) (leftOrdinal := 0)
    (rightResult := 19585) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63708

namespace LeftMerge63725
def owner : Owner := ⟨.program ⟨257⟩, ⟨34608⟩⟩
def mergeEvent : Nat := 63725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63719RawTerms
def rightRaw : List Term := Proof.Events009.exact2456RawTerms
def group : MergeGroup := .operator 63719 2456
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63719) (leftOrdinal := 1)
    (rightResult := 2456) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩, ⟨.program ⟨257⟩, ⟨34602⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63725

namespace LeftMerge63726
def owner : Owner := ⟨.program ⟨257⟩, ⟨34608⟩⟩
def mergeEvent : Nat := 63726
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events248.exact63719RawTerms
def rightRaw : List Term := Proof.Events009.exact2456RawTerms
def group : MergeGroup := .operator 63719 2456
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63719) (leftOrdinal := 0)
    (rightResult := 2456) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63726

namespace LeftMerge63731
def owner : Owner := ⟨.program ⟨257⟩, ⟨13687⟩⟩
def mergeEvent : Nat := 63731
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2456RawTerms
def rightRaw : List Term := Proof.Events239.exact61278RawTerms
def group : MergeGroup := .operator 2456 61278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2456) (leftOrdinal := 0)
    (rightResult := 61278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63731

namespace LeftMerge63736
def owner : Owner := ⟨.program ⟨257⟩, ⟨10779⟩⟩
def mergeEvent : Nat := 63736
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61148RawTerms
def rightRaw : List Term := Proof.Events076.exact19626RawTerms
def group : MergeGroup := .operator 61148 19626
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61148) (leftOrdinal := 0)
    (rightResult := 19626) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63736

namespace LeftMerge63753
def owner : Owner := ⟨.program ⟨257⟩, ⟨13690⟩⟩
def mergeEvent : Nat := 63753
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63747RawTerms
def rightRaw : List Term := Proof.Events076.exact19615RawTerms
def group : MergeGroup := .operator 63747 19615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63747) (leftOrdinal := 1)
    (rightResult := 19615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63753

namespace LeftMerge63755
def owner : Owner := ⟨.program ⟨257⟩, ⟨13690⟩⟩
def mergeEvent : Nat := 63755
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def rhsRaw : List Term := Proof.Events076.exact19585RawTerms
def group : MergeGroup := .relation 63754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 63754) (rhsResult := 19585)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9550⟩⟩) ⟨7280⟩ 19585) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge63755

namespace LeftMerge63756
def owner : Owner := ⟨.program ⟨257⟩, ⟨13690⟩⟩
def mergeEvent : Nat := 63756
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63747RawTerms
def rightRaw : List Term := Proof.Events076.exact19615RawTerms
def group : MergeGroup := .operator 63747 19615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63747) (leftOrdinal := 0)
    (rightResult := 19615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63756

namespace LeftMerge63761
def owner : Owner := ⟨.program ⟨257⟩, ⟨34609⟩⟩
def mergeEvent : Nat := 63761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events249.exact63757RawTerms
def rightRaw : List Term := Proof.Events248.exact63727RawTerms
def group : MergeGroup := .operator 63757 63727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 63757) (leftOrdinal := 1)
    (rightResult := 63727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨13686⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge63761

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
