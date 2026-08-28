import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge134476
def owner : Owner := ⟨.program ⟨257⟩, ⟨49583⟩⟩
def mergeEvent : Nat := 134476
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134467RawTerms
def rightRaw : List Term := Proof.Events524.exact134398RawTerms
def group : MergeGroup := .operator 134467 134398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134467) (leftOrdinal := 0)
    (rightResult := 134398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134476

namespace LeftMerge134488
def owner : Owner := ⟨.program ⟨257⟩, ⟨5472⟩⟩
def mergeEvent : Nat := 134488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }
def leftRaw : List Term := Proof.Events524.exact134273RawTerms
def rightRaw : List Term := Proof.Events067.exact17158RawTerms
def group : MergeGroup := .operator 134273 17158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134273) (leftOrdinal := 0)
    (rightResult := 17158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134488

namespace LeftMerge134501
def owner : Owner := ⟨.program ⟨257⟩, ⟨48522⟩⟩
def mergeEvent : Nat := 134501
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events525.exact134484RawTerms
def group : MergeGroup := .operator 134495 134484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 134484) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48519⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134501

namespace LeftMerge134580
def owner : Owner := ⟨.program ⟨257⟩, ⟨47667⟩⟩
def mergeEvent : Nat := 134580
def frameStart : Nat := 134550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events525.exact134576RawTerms
def rightRaw : List Term := Proof.Events525.exact134573RawTerms
def group : MergeGroup := .operator 134576 134573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134576) (leftOrdinal := 0)
    (rightResult := 134573) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14976⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134580

namespace LeftMerge134610
def owner : Owner := ⟨.program ⟨257⟩, ⟨49400⟩⟩
def mergeEvent : Nat := 134610
def frameStart : Nat := 134550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134606RawTerms
def rightRaw : List Term := Proof.Events525.exact134604RawTerms
def group : MergeGroup := .operator 134606 134604
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134606) (leftOrdinal := 0)
    (rightResult := 134604) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134610

namespace LeftMerge134633
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 134633
def frameStart : Nat := 134550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134629RawTerms
def rightRaw : List Term := Proof.Events525.exact134626RawTerms
def group : MergeGroup := .operator 134629 134626
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134629) (leftOrdinal := 0)
    (rightResult := 134626) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134633

namespace LeftMerge134642
def owner : Owner := ⟨.program ⟨257⟩, ⟨49585⟩⟩
def mergeEvent : Nat := 134642
def frameStart : Nat := 134550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134638RawTerms
def rightRaw : List Term := Proof.Events525.exact134595RawTerms
def group : MergeGroup := .operator 134638 134595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134638) (leftOrdinal := 0)
    (rightResult := 134595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49582⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134642

namespace LeftMerge134643
def owner : Owner := ⟨.program ⟨257⟩, ⟨49585⟩⟩
def mergeEvent : Nat := 134643
def frameStart : Nat := 134550
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134638RawTerms
def rightRaw : List Term := Proof.Events525.exact134595RawTerms
def group : MergeGroup := .operator 134638 134595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134638) (leftOrdinal := 1)
    (rightResult := 134595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge134643

namespace LeftMerge134645
def owner : Owner := ⟨.program ⟨257⟩, ⟨49585⟩⟩
def mergeEvent : Nat := 134645
def frameStart : Nat := 134550
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49107⟩⟩] } }
def rhsRaw : List Term := Proof.Events525.exact134592RawTerms
def group : MergeGroup := .relation 134644
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 134644) (rhsResult := 134592)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49582⟩⟩) ⟨49107⟩ 134592) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49107⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨49107⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge134645

namespace LeftMerge134653
def owner : Owner := ⟨.program ⟨257⟩, ⟨48094⟩⟩
def mergeEvent : Nat := 134653
def frameStart : Nat := 134550
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134606RawTerms
def rightRaw : List Term := Proof.Events525.exact134649RawTerms
def group : MergeGroup := .operator 134606 134649
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134606) (leftOrdinal := 0)
    (rightResult := 134649) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48092⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134653

namespace LeftMerge134670
def owner : Owner := ⟨.program ⟨257⟩, ⟨48522⟩⟩
def mergeEvent : Nat := 134670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events526.exact134667RawTerms
def group : MergeGroup := .relation 134669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 134669) (rhsResult := 134667)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 134668 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩) (none) 134667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134670

namespace LeftMerge134671
def owner : Owner := ⟨.program ⟨257⟩, ⟨48522⟩⟩
def mergeEvent : Nat := 134671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩] } }
def rhsRaw : List Term := Proof.Events526.exact134667RawTerms
def group : MergeGroup := .relation 134669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 134669) (rhsResult := 134667)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 134668 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩) (none) 134667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge134671

namespace LeftMerge134672
def owner : Owner := ⟨.program ⟨257⟩, ⟨48522⟩⟩
def mergeEvent : Nat := 134672
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49107⟩⟩] } }
def rhsRaw : List Term := Proof.Events526.exact134667RawTerms
def group : MergeGroup := .relation 134669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 134669) (rhsResult := 134667)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 134668 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩) (none) 134667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49107⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨49107⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134672

namespace LeftMerge134673
def owner : Owner := ⟨.program ⟨257⟩, ⟨48522⟩⟩
def mergeEvent : Nat := 134673
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events526.exact134667RawTerms
def group : MergeGroup := .relation 134669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 134669) (rhsResult := 134667)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 134668 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48519⟩⟩]⟩) (none) 134667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48092⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨48092⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge134673

namespace LeftMerge134678
def owner : Owner := ⟨.program ⟨257⟩, ⟨49584⟩⟩
def mergeEvent : Nat := 134678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49107⟩⟩] } }
def leftRaw : List Term := Proof.Events526.exact134674RawTerms
def rightRaw : List Term := Proof.Events525.exact134477RawTerms
def group : MergeGroup := .operator 134674 134477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134674) (leftOrdinal := 2)
    (rightResult := 134477) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49107⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49107⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨14976⟩⟩, ⟨.program ⟨257⟩, ⟨47666⟩⟩], [⟨.program ⟨257⟩, ⟨49107⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge134678

namespace LeftMerge134679
def owner : Owner := ⟨.program ⟨257⟩, ⟨49584⟩⟩
def mergeEvent : Nat := 134679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩] } }
def leftRaw : List Term := Proof.Events526.exact134674RawTerms
def rightRaw : List Term := Proof.Events525.exact134477RawTerms
def group : MergeGroup := .operator 134674 134477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134674) (leftOrdinal := 1)
    (rightResult := 134477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49582⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge134679

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
