import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge34688
def owner : Owner := ⟨.program ⟨257⟩, ⟨34822⟩⟩
def mergeEvent : Nat := 34688
def frameStart : Nat := 34585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events135.exact34641RawTerms
def rightRaw : List Term := Proof.Events135.exact34684RawTerms
def group : MergeGroup := .operator 34641 34684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34641) (leftOrdinal := 0)
    (rightResult := 34684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34688

namespace LeftMerge34705
def owner : Owner := ⟨.program ⟨257⟩, ⟨35282⟩⟩
def mergeEvent : Nat := 34705
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events135.exact34702RawTerms
def group : MergeGroup := .relation 34704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34704) (rhsResult := 34702)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 34703 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩) (none) 34702) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34705

namespace LeftMerge34706
def owner : Owner := ⟨.program ⟨257⟩, ⟨35282⟩⟩
def mergeEvent : Nat := 34706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩] } }
def rhsRaw : List Term := Proof.Events135.exact34702RawTerms
def group : MergeGroup := .relation 34704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34704) (rhsResult := 34702)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 34703 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩) (none) 34702) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34706

namespace LeftMerge34707
def owner : Owner := ⟨.program ⟨257⟩, ⟨35282⟩⟩
def mergeEvent : Nat := 34707
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35803⟩⟩] } }
def rhsRaw : List Term := Proof.Events135.exact34702RawTerms
def group : MergeGroup := .relation 34704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34704) (rhsResult := 34702)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 34703 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩) (none) 34702) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35803⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], [⟨.program ⟨257⟩, ⟨35803⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34707

namespace LeftMerge34708
def owner : Owner := ⟨.program ⟨257⟩, ⟨35282⟩⟩
def mergeEvent : Nat := 34708
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events135.exact34702RawTerms
def group : MergeGroup := .relation 34704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34704) (rhsResult := 34702)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 34703 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩) (none) 34702) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34708

namespace LeftMerge34713
def owner : Owner := ⟨.program ⟨257⟩, ⟨36360⟩⟩
def mergeEvent : Nat := 34713
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35803⟩⟩] } }
def leftRaw : List Term := Proof.Events135.exact34709RawTerms
def rightRaw : List Term := Proof.Events134.exact34523RawTerms
def group : MergeGroup := .operator 34709 34523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34709) (leftOrdinal := 2)
    (rightResult := 34523) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35803⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35803⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13716⟩⟩, ⟨.program ⟨257⟩, ⟨34650⟩⟩], [⟨.program ⟨257⟩, ⟨35803⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34713

namespace LeftMerge34714
def owner : Owner := ⟨.program ⟨257⟩, ⟨36360⟩⟩
def mergeEvent : Nat := 34714
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩] } }
def leftRaw : List Term := Proof.Events135.exact34709RawTerms
def rightRaw : List Term := Proof.Events134.exact34523RawTerms
def group : MergeGroup := .operator 34709 34523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34709) (leftOrdinal := 1)
    (rightResult := 34523) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36358⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34714

namespace LeftMerge34722
def owner : Owner := ⟨.program ⟨257⟩, ⟨36856⟩⟩
def mergeEvent : Nat := 34722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩] } }
def leftRaw : List Term := Proof.Events135.exact34716RawTerms
def rightRaw : List Term := Proof.Events134.exact34439RawTerms
def group : MergeGroup := .operator 34716 34439
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34716) (leftOrdinal := 0)
    (rightResult := 34439) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36854⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34722

namespace LeftMerge34723
def owner : Owner := ⟨.program ⟨257⟩, ⟨36856⟩⟩
def mergeEvent : Nat := 34723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩] } }
def leftRaw : List Term := Proof.Events135.exact34716RawTerms
def rightRaw : List Term := Proof.Events134.exact34439RawTerms
def group : MergeGroup := .operator 34716 34439
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34716) (leftOrdinal := 1)
    (rightResult := 34439) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36854⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34723

namespace LeftMerge34725
def owner : Owner := ⟨.program ⟨257⟩, ⟨36856⟩⟩
def mergeEvent : Nat := 34725
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35982⟩⟩] } }
def rhsRaw : List Term := Proof.Events134.exact34436RawTerms
def group : MergeGroup := .relation 34724
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34724) (rhsResult := 34436)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36854⟩⟩) ⟨35982⟩ 34436) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35982⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨35982⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34725

namespace LeftMerge34739
def owner : Owner := ⟨.program ⟨257⟩, ⟨35679⟩⟩
def mergeEvent : Nat := 34739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35676⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events135.exact34733RawTerms
def group : MergeGroup := .operator 32120 34733
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 34733) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35676⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35676⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34739

namespace LeftMerge34860
def owner : Owner := ⟨.program ⟨257⟩, ⟨36144⟩⟩
def mergeEvent : Nat := 34860
def frameStart : Nat := 34794
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events136.exact34856RawTerms
def rightRaw : List Term := Proof.Events136.exact34854RawTerms
def group : MergeGroup := .operator 34856 34854
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34856) (leftOrdinal := 0)
    (rightResult := 34854) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34860

namespace LeftMerge34872
def owner : Owner := ⟨.program ⟨257⟩, ⟨36855⟩⟩
def mergeEvent : Nat := 34872
def frameStart : Nat := 34794
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩] } }
def leftRaw : List Term := Proof.Events136.exact34868RawTerms
def rightRaw : List Term := Proof.Events136.exact34845RawTerms
def group : MergeGroup := .operator 34868 34845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34868) (leftOrdinal := 0)
    (rightResult := 34845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36854⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34872

namespace LeftMerge34873
def owner : Owner := ⟨.program ⟨257⟩, ⟨36855⟩⟩
def mergeEvent : Nat := 34873
def frameStart : Nat := 34794
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩] } }
def leftRaw : List Term := Proof.Events136.exact34868RawTerms
def rightRaw : List Term := Proof.Events136.exact34845RawTerms
def group : MergeGroup := .operator 34868 34845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34868) (leftOrdinal := 1)
    (rightResult := 34845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36854⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34873

namespace LeftMerge34875
def owner : Owner := ⟨.program ⟨257⟩, ⟨36855⟩⟩
def mergeEvent : Nat := 34875
def frameStart : Nat := 34794
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34820⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35982⟩⟩] } }
def rhsRaw : List Term := Proof.Events136.exact34842RawTerms
def group : MergeGroup := .relation 34874
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 34874) (rhsResult := 34842)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36854⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36854⟩⟩) ⟨35982⟩ 34842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35982⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34820⟩⟩], [⟨.program ⟨257⟩, ⟨35982⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge34875

namespace LeftMerge34883
def owner : Owner := ⟨.program ⟨257⟩, ⟨35081⟩⟩
def mergeEvent : Nat := 34883
def frameStart : Nat := 34794
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events136.exact34856RawTerms
def rightRaw : List Term := Proof.Events136.exact34879RawTerms
def group : MergeGroup := .operator 34856 34879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34856) (leftOrdinal := 0)
    (rightResult := 34879) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35080⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨35080⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge34883

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
