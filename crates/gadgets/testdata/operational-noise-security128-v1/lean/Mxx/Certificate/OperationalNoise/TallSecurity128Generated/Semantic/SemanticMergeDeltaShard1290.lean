import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge209696
def owner : Owner := ⟨.program ⟨257⟩, ⟨38942⟩⟩
def mergeEvent : Nat := 209696
def frameStart : Nat := 209603
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩] } }
def leftRaw : List Term := Proof.Events819.exact209691RawTerms
def rightRaw : List Term := Proof.Events818.exact209648RawTerms
def group : MergeGroup := .operator 209691 209648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209691) (leftOrdinal := 1)
    (rightResult := 209648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38939⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209696

namespace LeftMerge209698
def owner : Owner := ⟨.program ⟨257⟩, ⟨38942⟩⟩
def mergeEvent : Nat := 209698
def frameStart : Nat := 209603
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38429⟩⟩] } }
def rhsRaw : List Term := Proof.Events818.exact209645RawTerms
def group : MergeGroup := .relation 209697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209697) (rhsResult := 209645)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38939⟩⟩) ⟨38429⟩ 209645) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38429⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], [⟨.program ⟨257⟩, ⟨38429⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209698

namespace LeftMerge209706
def owner : Owner := ⟨.program ⟨257⟩, ⟨37430⟩⟩
def mergeEvent : Nat := 209706
def frameStart : Nat := 209603
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events818.exact209659RawTerms
def rightRaw : List Term := Proof.Events819.exact209702RawTerms
def group : MergeGroup := .operator 209659 209702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209659) (leftOrdinal := 0)
    (rightResult := 209702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209706

namespace LeftMerge209723
def owner : Owner := ⟨.program ⟨257⟩, ⟨37872⟩⟩
def mergeEvent : Nat := 209723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events819.exact209720RawTerms
def group : MergeGroup := .relation 209722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209722) (rhsResult := 209720)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209721 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37869⟩⟩]⟩) (none) 209720) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209723

namespace LeftMerge209724
def owner : Owner := ⟨.program ⟨257⟩, ⟨37872⟩⟩
def mergeEvent : Nat := 209724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩] } }
def rhsRaw : List Term := Proof.Events819.exact209720RawTerms
def group : MergeGroup := .relation 209722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209722) (rhsResult := 209720)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209721 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37869⟩⟩]⟩) (none) 209720) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209724

namespace LeftMerge209725
def owner : Owner := ⟨.program ⟨257⟩, ⟨37872⟩⟩
def mergeEvent : Nat := 209725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38429⟩⟩] } }
def rhsRaw : List Term := Proof.Events819.exact209720RawTerms
def group : MergeGroup := .relation 209722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209722) (rhsResult := 209720)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209721 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37869⟩⟩]⟩) (none) 209720) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38429⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], [⟨.program ⟨257⟩, ⟨38429⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209725

namespace LeftMerge209726
def owner : Owner := ⟨.program ⟨257⟩, ⟨37872⟩⟩
def mergeEvent : Nat := 209726
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events819.exact209720RawTerms
def group : MergeGroup := .relation 209722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209722) (rhsResult := 209720)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37869⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 209721 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37869⟩⟩]⟩) (none) 209720) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209726

namespace LeftMerge209731
def owner : Owner := ⟨.program ⟨257⟩, ⟨38941⟩⟩
def mergeEvent : Nat := 209731
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38429⟩⟩] } }
def leftRaw : List Term := Proof.Events819.exact209727RawTerms
def rightRaw : List Term := Proof.Events818.exact209541RawTerms
def group : MergeGroup := .operator 209727 209541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209727) (leftOrdinal := 2)
    (rightResult := 209541) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38429⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38429⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13881⟩⟩, ⟨.program ⟨257⟩, ⟨37114⟩⟩], [⟨.program ⟨257⟩, ⟨38429⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209731

namespace LeftMerge209732
def owner : Owner := ⟨.program ⟨257⟩, ⟨38941⟩⟩
def mergeEvent : Nat := 209732
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩] } }
def leftRaw : List Term := Proof.Events819.exact209727RawTerms
def rightRaw : List Term := Proof.Events818.exact209541RawTerms
def group : MergeGroup := .operator 209727 209541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209727) (leftOrdinal := 1)
    (rightResult := 209541) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38939⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209732

namespace LeftMerge209740
def owner : Owner := ⟨.program ⟨257⟩, ⟨39311⟩⟩
def mergeEvent : Nat := 209740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩] } }
def leftRaw : List Term := Proof.Events819.exact209734RawTerms
def rightRaw : List Term := Proof.Events818.exact209457RawTerms
def group : MergeGroup := .operator 209734 209457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209734) (leftOrdinal := 0)
    (rightResult := 209457) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39309⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209740

namespace LeftMerge209741
def owner : Owner := ⟨.program ⟨257⟩, ⟨39311⟩⟩
def mergeEvent : Nat := 209741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩] } }
def leftRaw : List Term := Proof.Events819.exact209734RawTerms
def rightRaw : List Term := Proof.Events818.exact209457RawTerms
def group : MergeGroup := .operator 209734 209457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209734) (leftOrdinal := 1)
    (rightResult := 209457) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39309⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209741

namespace LeftMerge209743
def owner : Owner := ⟨.program ⟨257⟩, ⟨39311⟩⟩
def mergeEvent : Nat := 209743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38581⟩⟩] } }
def rhsRaw : List Term := Proof.Events818.exact209454RawTerms
def group : MergeGroup := .relation 209742
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 209742) (rhsResult := 209454)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39309⟩⟩) ⟨38581⟩ 209454) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38581⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨37428⟩⟩], [⟨.program ⟨257⟩, ⟨38581⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209743

namespace LeftMerge209757
def owner : Owner := ⟨.program ⟨257⟩, ⟨38179⟩⟩
def mergeEvent : Nat := 209757
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38176⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events819.exact209751RawTerms
def group : MergeGroup := .operator 207620 209751
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 209751) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38176⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38176⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209757

namespace LeftMerge209878
def owner : Owner := ⟨.program ⟨257⟩, ⟨38788⟩⟩
def mergeEvent : Nat := 209878
def frameStart : Nat := 209812
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events819.exact209874RawTerms
def rightRaw : List Term := Proof.Events819.exact209872RawTerms
def group : MergeGroup := .operator 209874 209872
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209874) (leftOrdinal := 0)
    (rightResult := 209872) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209878

namespace LeftMerge209890
def owner : Owner := ⟨.program ⟨257⟩, ⟨39310⟩⟩
def mergeEvent : Nat := 209890
def frameStart : Nat := 209812
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩] } }
def leftRaw : List Term := Proof.Events819.exact209886RawTerms
def rightRaw : List Term := Proof.Events819.exact209863RawTerms
def group : MergeGroup := .operator 209886 209863
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209886) (leftOrdinal := 0)
    (rightResult := 209863) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39309⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge209890

namespace LeftMerge209891
def owner : Owner := ⟨.program ⟨257⟩, ⟨39310⟩⟩
def mergeEvent : Nat := 209891
def frameStart : Nat := 209812
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩] } }
def leftRaw : List Term := Proof.Events819.exact209886RawTerms
def rightRaw : List Term := Proof.Events819.exact209863RawTerms
def group : MergeGroup := .operator 209886 209863
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 209886) (leftOrdinal := 1)
    (rightResult := 209863) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37428⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39309⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37428⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39309⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge209891

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
