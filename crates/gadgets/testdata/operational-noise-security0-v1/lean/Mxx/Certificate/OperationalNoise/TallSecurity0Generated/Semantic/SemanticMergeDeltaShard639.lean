import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge103706
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103706

namespace LeftMerge103707
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103707
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103707

namespace LeftMerge103708
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103708
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18674⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103708

namespace LeftMerge103709
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103709
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 34) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18163⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18163⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103709

namespace LeftMerge103710
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103710
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 30) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17078⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17078⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103710

namespace LeftMerge103711
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103711
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 29) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16791⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16791⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103711

namespace LeftMerge103712
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103712
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 28) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16672⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16672⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103712

namespace LeftMerge103713
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103713
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 35) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18198⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18198⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103713

namespace LeftMerge103714
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103714
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 33) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17897⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17897⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103714

namespace LeftMerge103715
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103715
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 31) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17113⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨17113⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103715

namespace LeftMerge103716
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103716
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 27) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16301⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16301⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103716

namespace LeftMerge103717
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103717
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 36) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18303⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨18303⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103717

namespace LeftMerge103718
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103718
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 26) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16098⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨16098⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103718

namespace LeftMerge103719
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103719
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 25) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15979⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15979⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103719

namespace LeftMerge103720
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103720
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 24) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15860⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15860⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103720

namespace LeftMerge103721
def owner : Owner := ⟨.program ⟨214⟩, ⟨18551⟩⟩
def mergeEvent : Nat := 103721
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103687RawTerms
def group : MergeGroup := .relation 103689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103689) (rhsResult := 103687)
    (sourceTermOrdinal := 23) (source := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103688 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18548⟩⟩]⟩) (none) 103687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15741⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18612⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5506⟩⟩, ⟨.program ⟨214⟩, ⟨15741⟩⟩], [⟨.program ⟨214⟩, ⟨18612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103721

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
