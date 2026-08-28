import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge137502
def owner : Owner := ⟨.program ⟨257⟩, ⟨30340⟩⟩
def mergeEvent : Nat := 137502
def frameStart : Nat := 137442
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137498RawTerms
def rightRaw : List Term := Proof.Events537.exact137496RawTerms
def group : MergeGroup := .operator 137498 137496
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137498) (leftOrdinal := 0)
    (rightResult := 137496) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137502

namespace LeftMerge137525
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def mergeEvent : Nat := 137525
def frameStart : Nat := 137442
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137521RawTerms
def rightRaw : List Term := Proof.Events537.exact137518RawTerms
def group : MergeGroup := .operator 137521 137518
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137521) (leftOrdinal := 0)
    (rightResult := 137518) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137525

namespace LeftMerge137534
def owner : Owner := ⟨.program ⟨257⟩, ⟨30525⟩⟩
def mergeEvent : Nat := 137534
def frameStart : Nat := 137442
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137530RawTerms
def rightRaw : List Term := Proof.Events537.exact137487RawTerms
def group : MergeGroup := .operator 137530 137487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137530) (leftOrdinal := 0)
    (rightResult := 137487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30522⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137534

namespace LeftMerge137535
def owner : Owner := ⟨.program ⟨257⟩, ⟨30525⟩⟩
def mergeEvent : Nat := 137535
def frameStart : Nat := 137442
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137530RawTerms
def rightRaw : List Term := Proof.Events537.exact137487RawTerms
def group : MergeGroup := .operator 137530 137487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137530) (leftOrdinal := 1)
    (rightResult := 137487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30522⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137535

namespace LeftMerge137537
def owner : Owner := ⟨.program ⟨257⟩, ⟨30525⟩⟩
def mergeEvent : Nat := 137537
def frameStart : Nat := 137442
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30047⟩⟩] } }
def rhsRaw : List Term := Proof.Events537.exact137484RawTerms
def group : MergeGroup := .relation 137536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137536) (rhsResult := 137484)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30522⟩⟩) ⟨30047⟩ 137484) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30047⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], [⟨.program ⟨257⟩, ⟨30047⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137537

namespace LeftMerge137545
def owner : Owner := ⟨.program ⟨257⟩, ⟨29034⟩⟩
def mergeEvent : Nat := 137545
def frameStart : Nat := 137442
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137498RawTerms
def rightRaw : List Term := Proof.Events537.exact137541RawTerms
def group : MergeGroup := .operator 137498 137541
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137498) (leftOrdinal := 0)
    (rightResult := 137541) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137545

namespace LeftMerge137562
def owner : Owner := ⟨.program ⟨257⟩, ⟨29462⟩⟩
def mergeEvent : Nat := 137562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events537.exact137559RawTerms
def group : MergeGroup := .relation 137561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137561) (rhsResult := 137559)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 137560 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29459⟩⟩]⟩) (none) 137559) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137562

namespace LeftMerge137563
def owner : Owner := ⟨.program ⟨257⟩, ⟨29462⟩⟩
def mergeEvent : Nat := 137563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩] } }
def rhsRaw : List Term := Proof.Events537.exact137559RawTerms
def group : MergeGroup := .relation 137561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137561) (rhsResult := 137559)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 137560 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29459⟩⟩]⟩) (none) 137559) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137563

namespace LeftMerge137564
def owner : Owner := ⟨.program ⟨257⟩, ⟨29462⟩⟩
def mergeEvent : Nat := 137564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30047⟩⟩] } }
def rhsRaw : List Term := Proof.Events537.exact137559RawTerms
def group : MergeGroup := .relation 137561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137561) (rhsResult := 137559)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 137560 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29459⟩⟩]⟩) (none) 137559) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30047⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], [⟨.program ⟨257⟩, ⟨30047⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137564

namespace LeftMerge137565
def owner : Owner := ⟨.program ⟨257⟩, ⟨29462⟩⟩
def mergeEvent : Nat := 137565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events537.exact137559RawTerms
def group : MergeGroup := .relation 137561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137561) (rhsResult := 137559)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 137560 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29459⟩⟩]⟩) (none) 137559) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137565

namespace LeftMerge137570
def owner : Owner := ⟨.program ⟨257⟩, ⟨30524⟩⟩
def mergeEvent : Nat := 137570
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30047⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137566RawTerms
def rightRaw : List Term := Proof.Events536.exact137380RawTerms
def group : MergeGroup := .operator 137566 137380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137566) (leftOrdinal := 2)
    (rightResult := 137380) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30047⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30047⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨13176⟩⟩, ⟨.program ⟨257⟩, ⟨28606⟩⟩], [⟨.program ⟨257⟩, ⟨30047⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137570

namespace LeftMerge137571
def owner : Owner := ⟨.program ⟨257⟩, ⟨30524⟩⟩
def mergeEvent : Nat := 137571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137566RawTerms
def rightRaw : List Term := Proof.Events536.exact137380RawTerms
def group : MergeGroup := .operator 137566 137380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137566) (leftOrdinal := 1)
    (rightResult := 137380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30522⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137571

namespace LeftMerge137579
def owner : Owner := ⟨.program ⟨257⟩, ⟨30796⟩⟩
def mergeEvent : Nat := 137579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30794⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137573RawTerms
def rightRaw : List Term := Proof.Events536.exact137296RawTerms
def group : MergeGroup := .operator 137573 137296
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137573) (leftOrdinal := 0)
    (rightResult := 137296) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30794⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30794⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137579

namespace LeftMerge137580
def owner : Owner := ⟨.program ⟨257⟩, ⟨30796⟩⟩
def mergeEvent : Nat := 137580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30794⟩⟩] } }
def leftRaw : List Term := Proof.Events537.exact137573RawTerms
def rightRaw : List Term := Proof.Events536.exact137296RawTerms
def group : MergeGroup := .operator 137573 137296
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 137573) (leftOrdinal := 1)
    (rightResult := 137296) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30794⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137580

namespace LeftMerge137582
def owner : Owner := ⟨.program ⟨257⟩, ⟨30796⟩⟩
def mergeEvent : Nat := 137582
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30178⟩⟩] } }
def rhsRaw : List Term := Proof.Events536.exact137293RawTerms
def group : MergeGroup := .relation 137581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 137581) (rhsResult := 137293)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30794⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30794⟩⟩) ⟨30178⟩ 137293) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30178⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨29032⟩⟩], [⟨.program ⟨257⟩, ⟨30178⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge137582

namespace LeftMerge137596
def owner : Owner := ⟨.program ⟨257⟩, ⟨29699⟩⟩
def mergeEvent : Nat := 137596
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29696⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events537.exact137590RawTerms
def group : MergeGroup := .operator 134495 137590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 137590) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29696⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge137596

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
