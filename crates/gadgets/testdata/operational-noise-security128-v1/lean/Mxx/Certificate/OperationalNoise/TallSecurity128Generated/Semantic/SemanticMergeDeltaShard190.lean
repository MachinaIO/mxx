import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge34481
def owner : Owner := ⟨.program ⟨257⟩, ⟨13717⟩⟩
def mergeEvent : Nat := 34481
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events003.exact960RawTerms
def rightRaw : List Term := Proof.Events125.exact32028RawTerms
def group : MergeGroup := .operator 960 32028
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 960) (leftOrdinal := 0)
    (rightResult := 32028) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13716⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34481

namespace LeftMerge34486
def owner : Owner := ⟨.program ⟨257⟩, ⟨11630⟩⟩
def mergeEvent : Nat := 34486
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }
def leftRaw : List Term := Proof.Events124.exact31898RawTerms
def rightRaw : List Term := Proof.Events076.exact19626RawTerms
def group : MergeGroup := .operator 31898 19626
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31898) (leftOrdinal := 0)
    (rightResult := 19626) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34486

namespace LeftMerge34503
def owner : Owner := ⟨.program ⟨257⟩, ⟨13720⟩⟩
def mergeEvent : Nat := 34503
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events134.exact34497RawTerms
def rightRaw : List Term := Proof.Events076.exact19615RawTerms
def group : MergeGroup := .operator 34497 19615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34497) (leftOrdinal := 1)
    (rightResult := 19615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34503

namespace LeftMerge34505
def owner : Owner := ⟨.program ⟨257⟩, ⟨13720⟩⟩
def mergeEvent : Nat := 34505
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def rhsRaw : List Term := Proof.Events076.exact19585RawTerms
def group : MergeGroup := .relation 34504
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34504) (rhsResult := 19585)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9550⟩⟩) ⟨7280⟩ 19585) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34505

namespace LeftMerge34506
def owner : Owner := ⟨.program ⟨257⟩, ⟨13720⟩⟩
def mergeEvent : Nat := 34506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events134.exact34497RawTerms
def rightRaw : List Term := Proof.Events076.exact19615RawTerms
def group : MergeGroup := .operator 34497 19615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34497) (leftOrdinal := 0)
    (rightResult := 19615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34506

namespace LeftMerge34511
def owner : Owner := ⟨.program ⟨257⟩, ⟨34657⟩⟩
def mergeEvent : Nat := 34511
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }
def leftRaw : List Term := Proof.Events134.exact34507RawTerms
def rightRaw : List Term := Proof.Events134.exact34477RawTerms
def group : MergeGroup := .operator 34507 34477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34507) (leftOrdinal := 1)
    (rightResult := 34477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7280⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩], [⟨.program ⟨257⟩, ⟨7280⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34511

namespace LeftMerge34519
def owner : Owner := ⟨.program ⟨257⟩, ⟨36359⟩⟩
def mergeEvent : Nat := 34519
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩] } }
def leftRaw : List Term := Proof.Events134.exact34513RawTerms
def rightRaw : List Term := Proof.Events134.exact34449RawTerms
def group : MergeGroup := .operator 34513 34449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34513) (leftOrdinal := 1)
    (rightResult := 34449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36358⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34519

namespace LeftMerge34521
def owner : Owner := ⟨.program ⟨257⟩, ⟨36359⟩⟩
def mergeEvent : Nat := 34521
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35803⟩⟩] } }
def rhsRaw : List Term := Proof.Events134.exact34446RawTerms
def group : MergeGroup := .relation 34520
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34520) (rhsResult := 34446)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36358⟩⟩) ⟨35803⟩ 34446) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35803⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], [⟨.program ⟨257⟩, ⟨35803⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34521

namespace LeftMerge34522
def owner : Owner := ⟨.program ⟨257⟩, ⟨36359⟩⟩
def mergeEvent : Nat := 34522
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩] } }
def leftRaw : List Term := Proof.Events134.exact34513RawTerms
def rightRaw : List Term := Proof.Events134.exact34449RawTerms
def group : MergeGroup := .operator 34513 34449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34513) (leftOrdinal := 0)
    (rightResult := 34449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36358⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34522

namespace LeftMerge34536
def owner : Owner := ⟨.program ⟨257⟩, ⟨35282⟩⟩
def mergeEvent : Nat := 34536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events134.exact34530RawTerms
def group : MergeGroup := .operator 32120 34530
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 34530) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35279⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34536

namespace LeftMerge34615
def owner : Owner := ⟨.program ⟨257⟩, ⟨34651⟩⟩
def mergeEvent : Nat := 34615
def frameStart : Nat := 34585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events135.exact34611RawTerms
def rightRaw : List Term := Proof.Events135.exact34608RawTerms
def group : MergeGroup := .operator 34611 34608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34611) (leftOrdinal := 0)
    (rightResult := 34608) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13716⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34615

namespace LeftMerge34645
def owner : Owner := ⟨.program ⟨257⟩, ⟨36064⟩⟩
def mergeEvent : Nat := 34645
def frameStart : Nat := 34585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events135.exact34641RawTerms
def rightRaw : List Term := Proof.Events135.exact34639RawTerms
def group : MergeGroup := .operator 34641 34639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34641) (leftOrdinal := 0)
    (rightResult := 34639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34645

namespace LeftMerge34668
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 34668
def frameStart : Nat := 34585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events135.exact34664RawTerms
def rightRaw : List Term := Proof.Events135.exact34661RawTerms
def group : MergeGroup := .operator 34664 34661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34664) (leftOrdinal := 0)
    (rightResult := 34661) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34668

namespace LeftMerge34677
def owner : Owner := ⟨.program ⟨257⟩, ⟨36361⟩⟩
def mergeEvent : Nat := 34677
def frameStart : Nat := 34585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩] } }
def leftRaw : List Term := Proof.Events135.exact34673RawTerms
def rightRaw : List Term := Proof.Events135.exact34630RawTerms
def group : MergeGroup := .operator 34673 34630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34673) (leftOrdinal := 0)
    (rightResult := 34630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36358⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34677

namespace LeftMerge34678
def owner : Owner := ⟨.program ⟨257⟩, ⟨36361⟩⟩
def mergeEvent : Nat := 34678
def frameStart : Nat := 34585
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩] } }
def leftRaw : List Term := Proof.Events135.exact34673RawTerms
def rightRaw : List Term := Proof.Events135.exact34630RawTerms
def group : MergeGroup := .operator 34673 34630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34673) (leftOrdinal := 1)
    (rightResult := 34630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36358⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34678

namespace LeftMerge34680
def owner : Owner := ⟨.program ⟨257⟩, ⟨36361⟩⟩
def mergeEvent : Nat := 34680
def frameStart : Nat := 34585
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35803⟩⟩] } }
def rhsRaw : List Term := Proof.Events135.exact34627RawTerms
def group : MergeGroup := .relation 34679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34679) (rhsResult := 34627)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36358⟩⟩) ⟨35803⟩ 34627) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35803⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], [⟨.program ⟨257⟩, ⟨35803⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34680

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
