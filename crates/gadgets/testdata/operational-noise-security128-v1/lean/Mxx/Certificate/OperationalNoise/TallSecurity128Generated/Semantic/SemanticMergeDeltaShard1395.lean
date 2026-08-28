import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge226733
def owner : Owner := ⟨.program ⟨257⟩, ⟨64431⟩⟩
def mergeEvent : Nat := 226733
def frameStart : Nat := 226638
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63923⟩⟩] } }
def rhsRaw : List Term := Proof.Events885.exact226680RawTerms
def group : MergeGroup := .relation 226732
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226732) (rhsResult := 226680)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64428⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64428⟩⟩) ⟨63923⟩ 226680) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63923⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨63923⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226733

namespace LeftMerge226741
def owner : Owner := ⟨.program ⟨257⟩, ⟨62802⟩⟩
def mergeEvent : Nat := 226741
def frameStart : Nat := 226638
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events885.exact226694RawTerms
def rightRaw : List Term := Proof.Events885.exact226737RawTerms
def group : MergeGroup := .operator 226694 226737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226694) (leftOrdinal := 0)
    (rightResult := 226737) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226741

namespace LeftMerge226758
def owner : Owner := ⟨.program ⟨257⟩, ⟨63362⟩⟩
def mergeEvent : Nat := 226758
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events885.exact226755RawTerms
def group : MergeGroup := .relation 226757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226757) (rhsResult := 226755)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 226756 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63359⟩⟩]⟩) (none) 226755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226758

namespace LeftMerge226759
def owner : Owner := ⟨.program ⟨257⟩, ⟨63362⟩⟩
def mergeEvent : Nat := 226759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64428⟩⟩] } }
def rhsRaw : List Term := Proof.Events885.exact226755RawTerms
def group : MergeGroup := .relation 226757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226757) (rhsResult := 226755)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 226756 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63359⟩⟩]⟩) (none) 226755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64428⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64428⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226759

namespace LeftMerge226760
def owner : Owner := ⟨.program ⟨257⟩, ⟨63362⟩⟩
def mergeEvent : Nat := 226760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63923⟩⟩] } }
def rhsRaw : List Term := Proof.Events885.exact226755RawTerms
def group : MergeGroup := .relation 226757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226757) (rhsResult := 226755)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 226756 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63359⟩⟩]⟩) (none) 226755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63923⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨63923⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226760

namespace LeftMerge226761
def owner : Owner := ⟨.program ⟨257⟩, ⟨63362⟩⟩
def mergeEvent : Nat := 226761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events885.exact226755RawTerms
def group : MergeGroup := .relation 226757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226757) (rhsResult := 226755)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 226756 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63359⟩⟩]⟩) (none) 226755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226761

namespace LeftMerge226766
def owner : Owner := ⟨.program ⟨257⟩, ⟨64430⟩⟩
def mergeEvent : Nat := 226766
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63923⟩⟩] } }
def leftRaw : List Term := Proof.Events885.exact226762RawTerms
def rightRaw : List Term := Proof.Events885.exact226576RawTerms
def group : MergeGroup := .operator 226762 226576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226762) (leftOrdinal := 2)
    (rightResult := 226576) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63923⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63923⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25478⟩⟩, ⟨.program ⟨257⟩, ⟨62438⟩⟩], [⟨.program ⟨257⟩, ⟨63923⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226766

namespace LeftMerge226767
def owner : Owner := ⟨.program ⟨257⟩, ⟨64430⟩⟩
def mergeEvent : Nat := 226767
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64428⟩⟩] } }
def leftRaw : List Term := Proof.Events885.exact226762RawTerms
def rightRaw : List Term := Proof.Events885.exact226576RawTerms
def group : MergeGroup := .operator 226762 226576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226762) (leftOrdinal := 1)
    (rightResult := 226576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64428⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64428⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64428⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226767

namespace LeftMerge226775
def owner : Owner := ⟨.program ⟨257⟩, ⟨64843⟩⟩
def mergeEvent : Nat := 226775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩] } }
def leftRaw : List Term := Proof.Events885.exact226769RawTerms
def rightRaw : List Term := Proof.Events884.exact226492RawTerms
def group : MergeGroup := .operator 226769 226492
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226769) (leftOrdinal := 0)
    (rightResult := 226492) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64841⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226775

namespace LeftMerge226776
def owner : Owner := ⟨.program ⟨257⟩, ⟨64843⟩⟩
def mergeEvent : Nat := 226776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩] } }
def leftRaw : List Term := Proof.Events885.exact226769RawTerms
def rightRaw : List Term := Proof.Events884.exact226492RawTerms
def group : MergeGroup := .operator 226769 226492
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226769) (leftOrdinal := 1)
    (rightResult := 226492) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64841⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226776

namespace LeftMerge226778
def owner : Owner := ⟨.program ⟨257⟩, ⟨64843⟩⟩
def mergeEvent : Nat := 226778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64072⟩⟩] } }
def rhsRaw : List Term := Proof.Events884.exact226489RawTerms
def group : MergeGroup := .relation 226777
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226777) (rhsResult := 226489)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64841⟩⟩) ⟨64072⟩ 226489) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64072⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨62800⟩⟩], [⟨.program ⟨257⟩, ⟨64072⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226778

namespace LeftMerge226792
def owner : Owner := ⟨.program ⟨257⟩, ⟨63659⟩⟩
def mergeEvent : Nat := 226792
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63656⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events885.exact226786RawTerms
def group : MergeGroup := .operator 222245 226786
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 226786) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63656⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63656⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226792

namespace LeftMerge226913
def owner : Owner := ⟨.program ⟨257⟩, ⟨64284⟩⟩
def mergeEvent : Nat := 226913
def frameStart : Nat := 226847
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events886.exact226909RawTerms
def rightRaw : List Term := Proof.Events886.exact226907RawTerms
def group : MergeGroup := .operator 226909 226907
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226909) (leftOrdinal := 0)
    (rightResult := 226907) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226913

namespace LeftMerge226925
def owner : Owner := ⟨.program ⟨257⟩, ⟨64842⟩⟩
def mergeEvent : Nat := 226925
def frameStart : Nat := 226847
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩] } }
def leftRaw : List Term := Proof.Events886.exact226921RawTerms
def rightRaw : List Term := Proof.Events886.exact226898RawTerms
def group : MergeGroup := .operator 226921 226898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226921) (leftOrdinal := 0)
    (rightResult := 226898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64841⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226925

namespace LeftMerge226926
def owner : Owner := ⟨.program ⟨257⟩, ⟨64842⟩⟩
def mergeEvent : Nat := 226926
def frameStart : Nat := 226847
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩] } }
def leftRaw : List Term := Proof.Events886.exact226921RawTerms
def rightRaw : List Term := Proof.Events886.exact226898RawTerms
def group : MergeGroup := .operator 226921 226898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226921) (leftOrdinal := 1)
    (rightResult := 226898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64841⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226926

namespace LeftMerge226928
def owner : Owner := ⟨.program ⟨257⟩, ⟨64842⟩⟩
def mergeEvent : Nat := 226928
def frameStart : Nat := 226847
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64072⟩⟩] } }
def rhsRaw : List Term := Proof.Events886.exact226895RawTerms
def group : MergeGroup := .relation 226927
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226927) (rhsResult := 226895)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62800⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64841⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64841⟩⟩) ⟨64072⟩ 226895) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64072⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62800⟩⟩], [⟨.program ⟨257⟩, ⟨64072⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226928

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
