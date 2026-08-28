import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge33542
def owner : Owner := ⟨.program ⟨257⟩, ⟨14320⟩⟩
def mergeEvent : Nat := 33542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events130.exact33533RawTerms
def rightRaw : List Term := Proof.Events072.exact18613RawTerms
def group : MergeGroup := .operator 33533 18613
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33533) (leftOrdinal := 0)
    (rightResult := 18613) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33542

namespace LeftMerge33547
def owner : Owner := ⟨.program ⟨257⟩, ⟨40017⟩⟩
def mergeEvent : Nat := 33547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33543RawTerms
def rightRaw : List Term := Proof.Events130.exact33513RawTerms
def group : MergeGroup := .operator 33543 33513
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33543) (leftOrdinal := 1)
    (rightResult := 33513) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7282⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩], [⟨.program ⟨257⟩, ⟨7282⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33547

namespace LeftMerge33555
def owner : Owner := ⟨.program ⟨257⟩, ⟨41719⟩⟩
def mergeEvent : Nat := 33555
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33549RawTerms
def rightRaw : List Term := Proof.Events130.exact33485RawTerms
def group : MergeGroup := .operator 33549 33485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33549) (leftOrdinal := 1)
    (rightResult := 33485) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41718⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33555

namespace LeftMerge33557
def owner : Owner := ⟨.program ⟨257⟩, ⟨41719⟩⟩
def mergeEvent : Nat := 33557
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41163⟩⟩] } }
def rhsRaw : List Term := Proof.Events130.exact33482RawTerms
def group : MergeGroup := .relation 33556
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33556) (rhsResult := 33482)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41718⟩⟩) ⟨41163⟩ 33482) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41163⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], [⟨.program ⟨257⟩, ⟨41163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33557

namespace LeftMerge33558
def owner : Owner := ⟨.program ⟨257⟩, ⟨41719⟩⟩
def mergeEvent : Nat := 33558
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33549RawTerms
def rightRaw : List Term := Proof.Events130.exact33485RawTerms
def group : MergeGroup := .operator 33549 33485
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33549) (leftOrdinal := 0)
    (rightResult := 33485) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41718⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33558

namespace LeftMerge33572
def owner : Owner := ⟨.program ⟨257⟩, ⟨40642⟩⟩
def mergeEvent : Nat := 33572
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events131.exact33566RawTerms
def group : MergeGroup := .operator 32120 33566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 33566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨40639⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33572

namespace LeftMerge33651
def owner : Owner := ⟨.program ⟨257⟩, ⟨40011⟩⟩
def mergeEvent : Nat := 33651
def frameStart : Nat := 33621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events131.exact33647RawTerms
def rightRaw : List Term := Proof.Events131.exact33644RawTerms
def group : MergeGroup := .operator 33647 33644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33647) (leftOrdinal := 0)
    (rightResult := 33644) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14316⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33651

namespace LeftMerge33681
def owner : Owner := ⟨.program ⟨257⟩, ⟨41424⟩⟩
def mergeEvent : Nat := 33681
def frameStart : Nat := 33621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33677RawTerms
def rightRaw : List Term := Proof.Events131.exact33675RawTerms
def group : MergeGroup := .operator 33677 33675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33677) (leftOrdinal := 0)
    (rightResult := 33675) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33681

namespace LeftMerge33704
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def mergeEvent : Nat := 33704
def frameStart : Nat := 33621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33700RawTerms
def rightRaw : List Term := Proof.Events131.exact33697RawTerms
def group : MergeGroup := .operator 33700 33697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33700) (leftOrdinal := 0)
    (rightResult := 33697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9556⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33704

namespace LeftMerge33713
def owner : Owner := ⟨.program ⟨257⟩, ⟨41721⟩⟩
def mergeEvent : Nat := 33713
def frameStart : Nat := 33621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33709RawTerms
def rightRaw : List Term := Proof.Events131.exact33666RawTerms
def group : MergeGroup := .operator 33709 33666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33709) (leftOrdinal := 0)
    (rightResult := 33666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41718⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33713

namespace LeftMerge33714
def owner : Owner := ⟨.program ⟨257⟩, ⟨41721⟩⟩
def mergeEvent : Nat := 33714
def frameStart : Nat := 33621
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33709RawTerms
def rightRaw : List Term := Proof.Events131.exact33666RawTerms
def group : MergeGroup := .operator 33709 33666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33709) (leftOrdinal := 1)
    (rightResult := 33666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41718⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33714

namespace LeftMerge33716
def owner : Owner := ⟨.program ⟨257⟩, ⟨41721⟩⟩
def mergeEvent : Nat := 33716
def frameStart : Nat := 33621
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41163⟩⟩] } }
def rhsRaw : List Term := Proof.Events131.exact33663RawTerms
def group : MergeGroup := .relation 33715
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33715) (rhsResult := 33663)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨41718⟩⟩) ⟨41163⟩ 33663) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41163⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], [⟨.program ⟨257⟩, ⟨41163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33716

namespace LeftMerge33724
def owner : Owner := ⟨.program ⟨257⟩, ⟨40182⟩⟩
def mergeEvent : Nat := 33724
def frameStart : Nat := 33621
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events131.exact33677RawTerms
def rightRaw : List Term := Proof.Events131.exact33720RawTerms
def group : MergeGroup := .operator 33677 33720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 33677) (leftOrdinal := 0)
    (rightResult := 33720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40180⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33724

namespace LeftMerge33741
def owner : Owner := ⟨.program ⟨257⟩, ⟨40642⟩⟩
def mergeEvent : Nat := 33741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }
def rhsRaw : List Term := Proof.Events131.exact33738RawTerms
def group : MergeGroup := .relation 33740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33740) (rhsResult := 33738)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 33739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩]⟩) (none) 33738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33741

namespace LeftMerge33742
def owner : Owner := ⟨.program ⟨257⟩, ⟨40642⟩⟩
def mergeEvent : Nat := 33742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩] } }
def rhsRaw : List Term := Proof.Events131.exact33738RawTerms
def group : MergeGroup := .relation 33740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33740) (rhsResult := 33738)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 33739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩]⟩) (none) 33738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7299⟩⟩, ⟨.program ⟨257⟩, ⟨9556⟩⟩, ⟨.program ⟨257⟩, ⟨41718⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge33742

namespace LeftMerge33743
def owner : Owner := ⟨.program ⟨257⟩, ⟨40642⟩⟩
def mergeEvent : Nat := 33743
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41163⟩⟩] } }
def rhsRaw : List Term := Proof.Events131.exact33738RawTerms
def group : MergeGroup := .relation 33740
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 33740) (rhsResult := 33738)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 33739 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40639⟩⟩]⟩) (none) 33738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41163⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨14316⟩⟩, ⟨.program ⟨257⟩, ⟨40010⟩⟩], [⟨.program ⟨257⟩, ⟨41163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge33743

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
