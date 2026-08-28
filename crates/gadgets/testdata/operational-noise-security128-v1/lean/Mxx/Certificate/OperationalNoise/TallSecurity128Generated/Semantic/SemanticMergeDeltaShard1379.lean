import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge223806
def owner : Owner := ⟨.program ⟨257⟩, ⟨41384⟩⟩
def mergeEvent : Nat := 223806
def frameStart : Nat := 223746
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events874.exact223802RawTerms
def rightRaw : List Term := Proof.Events874.exact223800RawTerms
def group : MergeGroup := .operator 223802 223800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223802) (leftOrdinal := 0)
    (rightResult := 223800) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223806

namespace LeftMerge223829
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 223829
def frameStart : Nat := 223746
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events874.exact223825RawTerms
def rightRaw : List Term := Proof.Events874.exact223822RawTerms
def group : MergeGroup := .operator 223825 223822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223825) (leftOrdinal := 0)
    (rightResult := 223822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223829

namespace LeftMerge223838
def owner : Owner := ⟨.program ⟨257⟩, ⟨41611⟩⟩
def mergeEvent : Nat := 223838
def frameStart : Nat := 223746
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩] } }
def leftRaw : List Term := Proof.Events874.exact223834RawTerms
def rightRaw : List Term := Proof.Events874.exact223791RawTerms
def group : MergeGroup := .operator 223834 223791
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223834) (leftOrdinal := 0)
    (rightResult := 223791) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41608⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223838

namespace LeftMerge223839
def owner : Owner := ⟨.program ⟨257⟩, ⟨41611⟩⟩
def mergeEvent : Nat := 223839
def frameStart : Nat := 223746
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩] } }
def leftRaw : List Term := Proof.Events874.exact223834RawTerms
def rightRaw : List Term := Proof.Events874.exact223791RawTerms
def group : MergeGroup := .operator 223834 223791
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223834) (leftOrdinal := 1)
    (rightResult := 223791) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41608⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223839

namespace LeftMerge223841
def owner : Owner := ⟨.program ⟨257⟩, ⟨41611⟩⟩
def mergeEvent : Nat := 223841
def frameStart : Nat := 223746
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41103⟩⟩] } }
def rhsRaw : List Term := Proof.Events874.exact223788RawTerms
def group : MergeGroup := .relation 223840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223840) (rhsResult := 223788)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41608⟩⟩) ⟨41103⟩ 223788) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41103⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], [⟨.program ⟨257⟩, ⟨41103⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223841

namespace LeftMerge223849
def owner : Owner := ⟨.program ⟨257⟩, ⟨40102⟩⟩
def mergeEvent : Nat := 223849
def frameStart : Nat := 223746
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events874.exact223802RawTerms
def rightRaw : List Term := Proof.Events874.exact223845RawTerms
def group : MergeGroup := .operator 223802 223845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223802) (leftOrdinal := 0)
    (rightResult := 223845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223849

namespace LeftMerge223866
def owner : Owner := ⟨.program ⟨257⟩, ⟨40542⟩⟩
def mergeEvent : Nat := 223866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events874.exact223863RawTerms
def group : MergeGroup := .relation 223865
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223865) (rhsResult := 223863)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223864 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40539⟩⟩]⟩) (none) 223863) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223866

namespace LeftMerge223867
def owner : Owner := ⟨.program ⟨257⟩, ⟨40542⟩⟩
def mergeEvent : Nat := 223867
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩] } }
def rhsRaw : List Term := Proof.Events874.exact223863RawTerms
def group : MergeGroup := .relation 223865
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223865) (rhsResult := 223863)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223864 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40539⟩⟩]⟩) (none) 223863) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223867

namespace LeftMerge223868
def owner : Owner := ⟨.program ⟨257⟩, ⟨40542⟩⟩
def mergeEvent : Nat := 223868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41103⟩⟩] } }
def rhsRaw : List Term := Proof.Events874.exact223863RawTerms
def group : MergeGroup := .relation 223865
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223865) (rhsResult := 223863)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223864 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40539⟩⟩]⟩) (none) 223863) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41103⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], [⟨.program ⟨257⟩, ⟨41103⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223868

namespace LeftMerge223869
def owner : Owner := ⟨.program ⟨257⟩, ⟨40542⟩⟩
def mergeEvent : Nat := 223869
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events874.exact223863RawTerms
def group : MergeGroup := .relation 223865
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223865) (rhsResult := 223863)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40539⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 223864 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40539⟩⟩]⟩) (none) 223863) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223869

namespace LeftMerge223874
def owner : Owner := ⟨.program ⟨257⟩, ⟨41610⟩⟩
def mergeEvent : Nat := 223874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41103⟩⟩] } }
def leftRaw : List Term := Proof.Events874.exact223870RawTerms
def rightRaw : List Term := Proof.Events873.exact223684RawTerms
def group : MergeGroup := .operator 223870 223684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223870) (leftOrdinal := 2)
    (rightResult := 223684) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41103⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41103⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨14166⟩⟩, ⟨.program ⟨257⟩, ⟨39770⟩⟩], [⟨.program ⟨257⟩, ⟨41103⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223874

namespace LeftMerge223875
def owner : Owner := ⟨.program ⟨257⟩, ⟨41610⟩⟩
def mergeEvent : Nat := 223875
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩] } }
def leftRaw : List Term := Proof.Events874.exact223870RawTerms
def rightRaw : List Term := Proof.Events873.exact223684RawTerms
def group : MergeGroup := .operator 223870 223684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223870) (leftOrdinal := 1)
    (rightResult := 223684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41608⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223875

namespace LeftMerge223883
def owner : Owner := ⟨.program ⟨257⟩, ⟨41966⟩⟩
def mergeEvent : Nat := 223883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41964⟩⟩] } }
def leftRaw : List Term := Proof.Events874.exact223877RawTerms
def rightRaw : List Term := Proof.Events873.exact223600RawTerms
def group : MergeGroup := .operator 223877 223600
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223877) (leftOrdinal := 0)
    (rightResult := 223600) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41964⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨41964⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223883

namespace LeftMerge223884
def owner : Owner := ⟨.program ⟨257⟩, ⟨41966⟩⟩
def mergeEvent : Nat := 223884
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41964⟩⟩] } }
def leftRaw : List Term := Proof.Events874.exact223877RawTerms
def rightRaw : List Term := Proof.Events873.exact223600RawTerms
def group : MergeGroup := .operator 223877 223600
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 223877) (leftOrdinal := 1)
    (rightResult := 223600) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41964⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41964⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223884

namespace LeftMerge223886
def owner : Owner := ⟨.program ⟨257⟩, ⟨41966⟩⟩
def mergeEvent : Nat := 223886
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41252⟩⟩] } }
def rhsRaw : List Term := Proof.Events873.exact223597RawTerms
def group : MergeGroup := .relation 223885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 223885) (rhsResult := 223597)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41964⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41964⟩⟩) ⟨41252⟩ 223597) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41252⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨40100⟩⟩], [⟨.program ⟨257⟩, ⟨41252⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge223886

namespace LeftMerge223900
def owner : Owner := ⟨.program ⟨257⟩, ⟨40839⟩⟩
def mergeEvent : Nat := 223900
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40836⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events874.exact223894RawTerms
def group : MergeGroup := .operator 222245 223894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 223894) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40836⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40836⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge223900

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
