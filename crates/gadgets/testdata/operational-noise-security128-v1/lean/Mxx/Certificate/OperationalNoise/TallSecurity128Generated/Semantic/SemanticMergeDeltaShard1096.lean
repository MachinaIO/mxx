import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge178528
def owner : Owner := ⟨.program ⟨257⟩, ⟨48174⟩⟩
def mergeEvent : Nat := 178528
def frameStart : Nat := 178425
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events697.exact178481RawTerms
def rightRaw : List Term := Proof.Events697.exact178524RawTerms
def group : MergeGroup := .operator 178481 178524
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178481) (leftOrdinal := 0)
    (rightResult := 178524) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178528

namespace LeftMerge178545
def owner : Owner := ⟨.program ⟨257⟩, ⟨48622⟩⟩
def mergeEvent : Nat := 178545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events697.exact178542RawTerms
def group : MergeGroup := .relation 178544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178544) (rhsResult := 178542)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 178543 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48619⟩⟩]⟩) (none) 178542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178545

namespace LeftMerge178546
def owner : Owner := ⟨.program ⟨257⟩, ⟨48622⟩⟩
def mergeEvent : Nat := 178546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49692⟩⟩] } }
def rhsRaw : List Term := Proof.Events697.exact178542RawTerms
def group : MergeGroup := .relation 178544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178544) (rhsResult := 178542)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 178543 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48619⟩⟩]⟩) (none) 178542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49692⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49692⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178546

namespace LeftMerge178547
def owner : Owner := ⟨.program ⟨257⟩, ⟨48622⟩⟩
def mergeEvent : Nat := 178547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15126⟩⟩, ⟨.program ⟨257⟩, ⟨47906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49167⟩⟩] } }
def rhsRaw : List Term := Proof.Events697.exact178542RawTerms
def group : MergeGroup := .relation 178544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178544) (rhsResult := 178542)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 178543 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48619⟩⟩]⟩) (none) 178542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15126⟩⟩, ⟨.program ⟨257⟩, ⟨47906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49167⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15126⟩⟩, ⟨.program ⟨257⟩, ⟨47906⟩⟩], [⟨.program ⟨257⟩, ⟨49167⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178547

namespace LeftMerge178548
def owner : Owner := ⟨.program ⟨257⟩, ⟨48622⟩⟩
def mergeEvent : Nat := 178548
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events697.exact178542RawTerms
def group : MergeGroup := .relation 178544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178544) (rhsResult := 178542)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48619⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 178543 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48619⟩⟩]⟩) (none) 178542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178548

namespace LeftMerge178553
def owner : Owner := ⟨.program ⟨257⟩, ⟨49694⟩⟩
def mergeEvent : Nat := 178553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15126⟩⟩, ⟨.program ⟨257⟩, ⟨47906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49167⟩⟩] } }
def leftRaw : List Term := Proof.Events697.exact178549RawTerms
def rightRaw : List Term := Proof.Events696.exact178352RawTerms
def group : MergeGroup := .operator 178549 178352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178549) (leftOrdinal := 2)
    (rightResult := 178352) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15126⟩⟩, ⟨.program ⟨257⟩, ⟨47906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49167⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15126⟩⟩, ⟨.program ⟨257⟩, ⟨47906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49167⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15126⟩⟩, ⟨.program ⟨257⟩, ⟨47906⟩⟩], [⟨.program ⟨257⟩, ⟨49167⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178553

namespace LeftMerge178554
def owner : Owner := ⟨.program ⟨257⟩, ⟨49694⟩⟩
def mergeEvent : Nat := 178554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49692⟩⟩] } }
def leftRaw : List Term := Proof.Events697.exact178549RawTerms
def rightRaw : List Term := Proof.Events696.exact178352RawTerms
def group : MergeGroup := .operator 178549 178352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178549) (leftOrdinal := 1)
    (rightResult := 178352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49692⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49692⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49692⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178554

namespace LeftMerge178562
def owner : Owner := ⟨.program ⟨257⟩, ⟨50106⟩⟩
def mergeEvent : Nat := 178562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩] } }
def leftRaw : List Term := Proof.Events697.exact178556RawTerms
def rightRaw : List Term := Proof.Events696.exact178263RawTerms
def group : MergeGroup := .operator 178556 178263
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178556) (leftOrdinal := 0)
    (rightResult := 178263) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50104⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178562

namespace LeftMerge178563
def owner : Owner := ⟨.program ⟨257⟩, ⟨50106⟩⟩
def mergeEvent : Nat := 178563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩] } }
def leftRaw : List Term := Proof.Events697.exact178556RawTerms
def rightRaw : List Term := Proof.Events696.exact178263RawTerms
def group : MergeGroup := .operator 178556 178263
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178556) (leftOrdinal := 1)
    (rightResult := 178263) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50104⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178563

namespace LeftMerge178565
def owner : Owner := ⟨.program ⟨257⟩, ⟨50106⟩⟩
def mergeEvent : Nat := 178565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49328⟩⟩] } }
def rhsRaw : List Term := Proof.Events696.exact178260RawTerms
def group : MergeGroup := .relation 178564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178564) (rhsResult := 178260)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50104⟩⟩) ⟨49328⟩ 178260) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49328⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨48172⟩⟩], [⟨.program ⟨257⟩, ⟨49328⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178565

namespace LeftMerge178579
def owner : Owner := ⟨.program ⟨257⟩, ⟨48959⟩⟩
def mergeEvent : Nat := 178579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48956⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events697.exact178573RawTerms
def group : MergeGroup := .operator 178370 178573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 178573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48956⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48956⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178579

namespace LeftMerge178700
def owner : Owner := ⟨.program ⟨257⟩, ⟨49520⟩⟩
def mergeEvent : Nat := 178700
def frameStart : Nat := 178634
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events698.exact178696RawTerms
def rightRaw : List Term := Proof.Events698.exact178694RawTerms
def group : MergeGroup := .operator 178696 178694
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178696) (leftOrdinal := 0)
    (rightResult := 178694) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178700

namespace LeftMerge178712
def owner : Owner := ⟨.program ⟨257⟩, ⟨50105⟩⟩
def mergeEvent : Nat := 178712
def frameStart : Nat := 178634
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩] } }
def leftRaw : List Term := Proof.Events698.exact178708RawTerms
def rightRaw : List Term := Proof.Events697.exact178685RawTerms
def group : MergeGroup := .operator 178708 178685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178708) (leftOrdinal := 0)
    (rightResult := 178685) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50104⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178712

namespace LeftMerge178713
def owner : Owner := ⟨.program ⟨257⟩, ⟨50105⟩⟩
def mergeEvent : Nat := 178713
def frameStart : Nat := 178634
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩] } }
def leftRaw : List Term := Proof.Events698.exact178708RawTerms
def rightRaw : List Term := Proof.Events697.exact178685RawTerms
def group : MergeGroup := .operator 178708 178685
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178708) (leftOrdinal := 1)
    (rightResult := 178685) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50104⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178713

namespace LeftMerge178715
def owner : Owner := ⟨.program ⟨257⟩, ⟨50105⟩⟩
def mergeEvent : Nat := 178715
def frameStart : Nat := 178634
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48172⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49328⟩⟩] } }
def rhsRaw : List Term := Proof.Events697.exact178682RawTerms
def group : MergeGroup := .relation 178714
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 178714) (rhsResult := 178682)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48172⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50104⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50104⟩⟩) ⟨49328⟩ 178682) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49328⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48172⟩⟩], [⟨.program ⟨257⟩, ⟨49328⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge178715

namespace LeftMerge178723
def owner : Owner := ⟨.program ⟨257⟩, ⟨48403⟩⟩
def mergeEvent : Nat := 178723
def frameStart : Nat := 178634
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48402⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events698.exact178696RawTerms
def rightRaw : List Term := Proof.Events698.exact178719RawTerms
def group : MergeGroup := .operator 178696 178719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178696) (leftOrdinal := 0)
    (rightResult := 178719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48402⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48402⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge178723

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
