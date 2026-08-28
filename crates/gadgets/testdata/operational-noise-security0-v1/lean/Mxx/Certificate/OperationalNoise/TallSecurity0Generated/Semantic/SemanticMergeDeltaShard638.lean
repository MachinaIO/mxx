import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge103690
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103690
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 18) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103690

namespace LeftMerge103691
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103691
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 17) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103691

namespace LeftMerge103692
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103692
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 16) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103692

namespace LeftMerge103693
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103693
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 15) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103693

namespace LeftMerge103694
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103694
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 14) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103694

namespace LeftMerge103695
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 13) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103695

namespace LeftMerge103696
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103696
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 12) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103696

namespace LeftMerge103697
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103697
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 11) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103697

namespace LeftMerge103698
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103698
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 10) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103698

namespace LeftMerge103699
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103699
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 9) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103699

namespace LeftMerge103700
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103700
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 8) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103700

namespace LeftMerge103701
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103701
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 7) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103701

namespace LeftMerge103702
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103702
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 6) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103702

namespace LeftMerge103703
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 5) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103703

namespace LeftMerge103704
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 4) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103704

namespace LeftMerge103705
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103705
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103705

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
