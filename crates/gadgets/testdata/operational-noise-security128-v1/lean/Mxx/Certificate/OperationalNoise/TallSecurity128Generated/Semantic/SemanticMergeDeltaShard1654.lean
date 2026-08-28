import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge267542
def owner : Owner := ⟨.program ⟨257⟩, ⟨14060⟩⟩
def mergeEvent : Nat := 267542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267533RawTerms
def rightRaw : List Term := Proof.Events072.exact18613RawTerms
def group : MergeGroup := .operator 267533 18613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267533) (leftOrdinal := 0)
    (rightResult := 18613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267542

namespace LeftMerge267547
def owner : Owner := ⟨.program ⟨257⟩, ⟨39601⟩⟩
def mergeEvent : Nat := 267547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267543RawTerms
def rightRaw : List Term := Proof.Events1044.exact267513RawTerms
def group : MergeGroup := .operator 267543 267513
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267543) (leftOrdinal := 1)
    (rightResult := 267513) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267547

namespace LeftMerge267555
def owner : Owner := ⟨.program ⟨257⟩, ⟨41529⟩⟩
def mergeEvent : Nat := 267555
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267549RawTerms
def rightRaw : List Term := Proof.Events1044.exact267485RawTerms
def group : MergeGroup := .operator 267549 267485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267549) (leftOrdinal := 1)
    (rightResult := 267485) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41528⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267555

namespace LeftMerge267557
def owner : Owner := ⟨.program ⟨257⟩, ⟨41529⟩⟩
def mergeEvent : Nat := 267557
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41059⟩⟩] } }
def rhsRaw : List Term := Proof.Events1044.exact267482RawTerms
def group : MergeGroup := .relation 267556
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267556) (rhsResult := 267482)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41528⟩⟩) ⟨41059⟩ 267482) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41059⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], [⟨.program ⟨257⟩, ⟨41059⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267557

namespace LeftMerge267558
def owner : Owner := ⟨.program ⟨257⟩, ⟨41529⟩⟩
def mergeEvent : Nat := 267558
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267549RawTerms
def rightRaw : List Term := Proof.Events1044.exact267485RawTerms
def group : MergeGroup := .operator 267549 267485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267549) (leftOrdinal := 0)
    (rightResult := 267485) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41528⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267558

namespace LeftMerge267572
def owner : Owner := ⟨.program ⟨257⟩, ⟨40469⟩⟩
def mergeEvent : Nat := 267572
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40466⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1045.exact267566RawTerms
def group : MergeGroup := .operator 266120 267566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 267566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40466⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40466⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267572

namespace LeftMerge267651
def owner : Owner := ⟨.program ⟨257⟩, ⟨39595⟩⟩
def mergeEvent : Nat := 267651
def frameStart : Nat := 267621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1045.exact267647RawTerms
def rightRaw : List Term := Proof.Events1045.exact267644RawTerms
def group : MergeGroup := .operator 267647 267644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267647) (leftOrdinal := 0)
    (rightResult := 267644) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14056⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267651

namespace LeftMerge267681
def owner : Owner := ⟨.program ⟨257⟩, ⟨41356⟩⟩
def mergeEvent : Nat := 267681
def frameStart : Nat := 267621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267677RawTerms
def rightRaw : List Term := Proof.Events1045.exact267675RawTerms
def group : MergeGroup := .operator 267677 267675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267677) (leftOrdinal := 0)
    (rightResult := 267675) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267681

namespace LeftMerge267704
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 267704
def frameStart : Nat := 267621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267700RawTerms
def rightRaw : List Term := Proof.Events1045.exact267697RawTerms
def group : MergeGroup := .operator 267700 267697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267700) (leftOrdinal := 0)
    (rightResult := 267697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267704

namespace LeftMerge267713
def owner : Owner := ⟨.program ⟨257⟩, ⟨41531⟩⟩
def mergeEvent : Nat := 267713
def frameStart : Nat := 267621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267709RawTerms
def rightRaw : List Term := Proof.Events1045.exact267666RawTerms
def group : MergeGroup := .operator 267709 267666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267709) (leftOrdinal := 0)
    (rightResult := 267666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41528⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267713

namespace LeftMerge267714
def owner : Owner := ⟨.program ⟨257⟩, ⟨41531⟩⟩
def mergeEvent : Nat := 267714
def frameStart : Nat := 267621
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267709RawTerms
def rightRaw : List Term := Proof.Events1045.exact267666RawTerms
def group : MergeGroup := .operator 267709 267666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267709) (leftOrdinal := 1)
    (rightResult := 267666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41528⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267714

namespace LeftMerge267716
def owner : Owner := ⟨.program ⟨257⟩, ⟨41531⟩⟩
def mergeEvent : Nat := 267716
def frameStart : Nat := 267621
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41059⟩⟩] } }
def rhsRaw : List Term := Proof.Events1045.exact267663RawTerms
def group : MergeGroup := .relation 267715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267715) (rhsResult := 267663)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41528⟩⟩) ⟨41059⟩ 267663) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41059⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], [⟨.program ⟨257⟩, ⟨41059⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267716

namespace LeftMerge267724
def owner : Owner := ⟨.program ⟨257⟩, ⟨40044⟩⟩
def mergeEvent : Nat := 267724
def frameStart : Nat := 267621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1045.exact267677RawTerms
def rightRaw : List Term := Proof.Events1045.exact267720RawTerms
def group : MergeGroup := .operator 267677 267720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 267677) (leftOrdinal := 0)
    (rightResult := 267720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40042⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40042⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267724

namespace LeftMerge267741
def owner : Owner := ⟨.program ⟨257⟩, ⟨40469⟩⟩
def mergeEvent : Nat := 267741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events1045.exact267738RawTerms
def group : MergeGroup := .relation 267740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267740) (rhsResult := 267738)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40466⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 267739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40466⟩⟩]⟩) (none) 267738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267741

namespace LeftMerge267742
def owner : Owner := ⟨.program ⟨257⟩, ⟨40469⟩⟩
def mergeEvent : Nat := 267742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩] } }
def rhsRaw : List Term := Proof.Events1045.exact267738RawTerms
def group : MergeGroup := .relation 267740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267740) (rhsResult := 267738)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40466⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 267739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40466⟩⟩]⟩) (none) 267738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41528⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge267742

namespace LeftMerge267743
def owner : Owner := ⟨.program ⟨257⟩, ⟨40469⟩⟩
def mergeEvent : Nat := 267743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41059⟩⟩] } }
def rhsRaw : List Term := Proof.Events1045.exact267738RawTerms
def group : MergeGroup := .relation 267740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 267740) (rhsResult := 267738)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40466⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 267739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40466⟩⟩]⟩) (none) 267738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41059⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨14056⟩⟩, ⟨.program ⟨257⟩, ⟨39594⟩⟩], [⟨.program ⟨257⟩, ⟨41059⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge267743

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
