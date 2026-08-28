import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge249785
def owner : Owner := ⟨.program ⟨257⟩, ⟨55864⟩⟩
def mergeEvent : Nat := 249785
def frameStart : Nat := 249704
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55122⟩⟩] } }
def rhsRaw : List Term := Proof.Events975.exact249752RawTerms
def group : MergeGroup := .relation 249784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 249784) (rhsResult := 249752)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55863⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55863⟩⟩) ⟨55122⟩ 249752) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55122⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨55122⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge249785

namespace LeftMerge249793
def owner : Owner := ⟨.program ⟨257⟩, ⟨54110⟩⟩
def mergeEvent : Nat := 249793
def frameStart : Nat := 249704
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events975.exact249766RawTerms
def rightRaw : List Term := Proof.Events975.exact249789RawTerms
def group : MergeGroup := .operator 249766 249789
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 249766) (leftOrdinal := 0)
    (rightResult := 249789) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54107⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge249793

namespace LeftMerge249810
def owner : Owner := ⟨.program ⟨257⟩, ⟨54695⟩⟩
def mergeEvent : Nat := 249810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }
def rhsRaw : List Term := Proof.Events975.exact249807RawTerms
def group : MergeGroup := .relation 249809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 249809) (rhsResult := 249807)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 249808 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54692⟩⟩]⟩) (none) 249807) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge249810

namespace LeftMerge249811
def owner : Owner := ⟨.program ⟨257⟩, ⟨54695⟩⟩
def mergeEvent : Nat := 249811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55863⟩⟩] } }
def rhsRaw : List Term := Proof.Events975.exact249807RawTerms
def group : MergeGroup := .relation 249809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 249809) (rhsResult := 249807)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 249808 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54692⟩⟩]⟩) (none) 249807) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55863⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55863⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge249811

namespace LeftMerge249812
def owner : Owner := ⟨.program ⟨257⟩, ⟨54695⟩⟩
def mergeEvent : Nat := 249812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55122⟩⟩] } }
def rhsRaw : List Term := Proof.Events975.exact249807RawTerms
def group : MergeGroup := .relation 249809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 249809) (rhsResult := 249807)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 249808 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54692⟩⟩]⟩) (none) 249807) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55122⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨55122⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge249812

namespace LeftMerge249813
def owner : Owner := ⟨.program ⟨257⟩, ⟨54695⟩⟩
def mergeEvent : Nat := 249813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events975.exact249807RawTerms
def group : MergeGroup := .relation 249809
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 249809) (rhsResult := 249807)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54692⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 249808 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54692⟩⟩]⟩) (none) 249807) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge249813

namespace LeftMerge249818
def owner : Owner := ⟨.program ⟨257⟩, ⟨55866⟩⟩
def mergeEvent : Nat := 249818
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55863⟩⟩] } }
def leftRaw : List Term := Proof.Events975.exact249814RawTerms
def rightRaw : List Term := Proof.Events975.exact249636RawTerms
def group : MergeGroup := .operator 249814 249636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 249814) (leftOrdinal := 0)
    (rightResult := 249636) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55863⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55863⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55863⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge249818

namespace LeftMerge249819
def owner : Owner := ⟨.program ⟨257⟩, ⟨55866⟩⟩
def mergeEvent : Nat := 249819
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55122⟩⟩] } }
def leftRaw : List Term := Proof.Events975.exact249814RawTerms
def rightRaw : List Term := Proof.Events975.exact249636RawTerms
def group : MergeGroup := .operator 249814 249636
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 249814) (leftOrdinal := 2)
    (rightResult := 249636) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55122⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55122⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨53852⟩⟩], [⟨.program ⟨257⟩, ⟨55122⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge249819

namespace LeftMerge249827
def owner : Owner := ⟨.program ⟨257⟩, ⟨55867⟩⟩
def mergeEvent : Nat := 249827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }
def leftRaw : List Term := Proof.Events975.exact249821RawTerms
def rightRaw : List Term := Proof.Events061.exact15782RawTerms
def group : MergeGroup := .operator 249821 15782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 249821) (leftOrdinal := 0)
    (rightResult := 15782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge249827

namespace LeftMerge249828
def owner : Owner := ⟨.program ⟨257⟩, ⟨55867⟩⟩
def mergeEvent : Nat := 249828
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }
def leftRaw : List Term := Proof.Events975.exact249821RawTerms
def rightRaw : List Term := Proof.Events061.exact15782RawTerms
def group : MergeGroup := .operator 249821 15782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 249821) (leftOrdinal := 1)
    (rightResult := 15782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge249828

namespace LeftMerge249830
def owner : Owner := ⟨.program ⟨257⟩, ⟨55867⟩⟩
def mergeEvent : Nat := 249830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15775RawTerms
def group : MergeGroup := .relation 249829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 249829) (rhsResult := 15775)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7125⟩⟩) ⟨7028⟩ 15775) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54107⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge249830

namespace LeftMerge249844
def owner : Owner := ⟨.program ⟨257⟩, ⟨52885⟩⟩
def mergeEvent : Nat := 249844
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52883⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243322RawTerms
def rightRaw : List Term := Proof.Events975.exact249838RawTerms
def group : MergeGroup := .operator 243322 249838
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243322) (leftOrdinal := 0)
    (rightResult := 249838) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52883⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52883⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge249844

namespace LeftMerge249845
def owner : Owner := ⟨.program ⟨257⟩, ⟨52885⟩⟩
def mergeEvent : Nat := 249845
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52883⟩⟩] } }
def leftRaw : List Term := Proof.Events950.exact243322RawTerms
def rightRaw : List Term := Proof.Events975.exact249838RawTerms
def group : MergeGroup := .operator 243322 249838
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 243322) (leftOrdinal := 1)
    (rightResult := 249838) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52883⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52883⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge249845

namespace LeftMerge249847
def owner : Owner := ⟨.program ⟨257⟩, ⟨52885⟩⟩
def mergeEvent : Nat := 249847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52142⟩⟩] } }
def rhsRaw : List Term := Proof.Events975.exact249835RawTerms
def group : MergeGroup := .relation 249846
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 249846) (rhsResult := 249835)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52883⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52883⟩⟩) ⟨52142⟩ 249835) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52142⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨52142⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge249847

namespace LeftMerge249861
def owner : Owner := ⟨.program ⟨257⟩, ⟨51715⟩⟩
def mergeEvent : Nat := 249861
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51712⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events975.exact249855RawTerms
def group : MergeGroup := .operator 236870 249855
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 249855) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51712⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51712⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge249861

namespace LeftMerge249982
def owner : Owner := ⟨.program ⟨257⟩, ⟨52360⟩⟩
def mergeEvent : Nat := 249982
def frameStart : Nat := 249916
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events976.exact249978RawTerms
def rightRaw : List Term := Proof.Events976.exact249976RawTerms
def group : MergeGroup := .operator 249978 249976
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 249978) (leftOrdinal := 0)
    (rightResult := 249976) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50872⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge249982

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
