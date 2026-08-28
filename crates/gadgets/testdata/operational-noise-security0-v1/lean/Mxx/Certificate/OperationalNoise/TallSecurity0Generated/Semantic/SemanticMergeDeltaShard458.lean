import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge75520
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75520
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75520

namespace LeftMerge75521
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75521
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75521

namespace LeftMerge75522
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75522
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 34) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18167⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75522

namespace LeftMerge75523
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75523
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 30) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17082⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75523

namespace LeftMerge75524
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 29) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75524

namespace LeftMerge75525
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 28) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16676⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16676⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75525

namespace LeftMerge75526
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75526
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 35) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18202⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75526

namespace LeftMerge75527
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75527
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 33) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17901⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17901⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75527

namespace LeftMerge75528
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 31) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17117⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨17117⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75528

namespace LeftMerge75529
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 27) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16305⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16305⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75529

namespace LeftMerge75530
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75530
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 36) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18327⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨18327⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75530

namespace LeftMerge75531
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75531
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 26) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16102⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨16102⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75531

namespace LeftMerge75532
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75532
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 25) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15983⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15983⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75532

namespace LeftMerge75533
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75533
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 24) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75533

namespace LeftMerge75534
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75534
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 23) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75534

namespace LeftMerge75535
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 22) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩, ⟨.program ⟨214⟩, ⟨15626⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75535

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
