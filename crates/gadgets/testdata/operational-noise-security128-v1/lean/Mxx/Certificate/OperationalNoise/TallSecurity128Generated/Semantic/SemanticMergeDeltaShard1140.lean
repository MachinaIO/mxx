import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge186553
def owner : Owner := ⟨.program ⟨257⟩, ⟨17393⟩⟩
def mergeEvent : Nat := 186553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩] } }
def leftRaw : List Term := Proof.Events728.exact186547RawTerms
def rightRaw : List Term := Proof.Events728.exact186483RawTerms
def group : MergeGroup := .operator 186547 186483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186547) (leftOrdinal := 1)
    (rightResult := 186483) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17392⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186553

namespace LeftMerge186555
def owner : Owner := ⟨.program ⟨257⟩, ⟨17393⟩⟩
def mergeEvent : Nat := 186555
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16867⟩⟩] } }
def rhsRaw : List Term := Proof.Events728.exact186480RawTerms
def group : MergeGroup := .relation 186554
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 186554) (rhsResult := 186480)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17392⟩⟩) ⟨16867⟩ 186480) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16867⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], [⟨.program ⟨257⟩, ⟨16867⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186555

namespace LeftMerge186556
def owner : Owner := ⟨.program ⟨257⟩, ⟨17393⟩⟩
def mergeEvent : Nat := 186556
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩] } }
def leftRaw : List Term := Proof.Events728.exact186547RawTerms
def rightRaw : List Term := Proof.Events728.exact186483RawTerms
def group : MergeGroup := .operator 186547 186483
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186547) (leftOrdinal := 0)
    (rightResult := 186483) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17392⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186556

namespace LeftMerge186570
def owner : Owner := ⟨.program ⟨257⟩, ⟨16322⟩⟩
def mergeEvent : Nat := 186570
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events728.exact186564RawTerms
def group : MergeGroup := .operator 178370 186564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 186564) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16319⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186570

namespace LeftMerge186649
def owner : Owner := ⟨.program ⟨257⟩, ⟨15547⟩⟩
def mergeEvent : Nat := 186649
def frameStart : Nat := 186619
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events729.exact186645RawTerms
def rightRaw : List Term := Proof.Events729.exact186642RawTerms
def group : MergeGroup := .operator 186645 186642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186645) (leftOrdinal := 0)
    (rightResult := 186642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12426⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186649

namespace LeftMerge186679
def owner : Owner := ⟨.program ⟨257⟩, ⟨17140⟩⟩
def mergeEvent : Nat := 186679
def frameStart : Nat := 186619
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events729.exact186675RawTerms
def rightRaw : List Term := Proof.Events729.exact186673RawTerms
def group : MergeGroup := .operator 186675 186673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186675) (leftOrdinal := 0)
    (rightResult := 186673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186679

namespace LeftMerge186702
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def mergeEvent : Nat := 186702
def frameStart : Nat := 186619
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }
def leftRaw : List Term := Proof.Events729.exact186698RawTerms
def rightRaw : List Term := Proof.Events729.exact186695RawTerms
def group : MergeGroup := .operator 186698 186695
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186698) (leftOrdinal := 0)
    (rightResult := 186695) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9568⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186702

namespace LeftMerge186711
def owner : Owner := ⟨.program ⟨257⟩, ⟨17395⟩⟩
def mergeEvent : Nat := 186711
def frameStart : Nat := 186619
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩] } }
def leftRaw : List Term := Proof.Events729.exact186707RawTerms
def rightRaw : List Term := Proof.Events729.exact186664RawTerms
def group : MergeGroup := .operator 186707 186664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186707) (leftOrdinal := 0)
    (rightResult := 186664) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17392⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186711

namespace LeftMerge186712
def owner : Owner := ⟨.program ⟨257⟩, ⟨17395⟩⟩
def mergeEvent : Nat := 186712
def frameStart : Nat := 186619
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩] } }
def leftRaw : List Term := Proof.Events729.exact186707RawTerms
def rightRaw : List Term := Proof.Events729.exact186664RawTerms
def group : MergeGroup := .operator 186707 186664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186707) (leftOrdinal := 1)
    (rightResult := 186664) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17392⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186712

namespace LeftMerge186714
def owner : Owner := ⟨.program ⟨257⟩, ⟨17395⟩⟩
def mergeEvent : Nat := 186714
def frameStart : Nat := 186619
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16867⟩⟩] } }
def rhsRaw : List Term := Proof.Events729.exact186661RawTerms
def group : MergeGroup := .relation 186713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 186713) (rhsResult := 186661)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17392⟩⟩) ⟨16867⟩ 186661) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16867⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], [⟨.program ⟨257⟩, ⟨16867⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186714

namespace LeftMerge186722
def owner : Owner := ⟨.program ⟨257⟩, ⟨15814⟩⟩
def mergeEvent : Nat := 186722
def frameStart : Nat := 186619
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events729.exact186675RawTerms
def rightRaw : List Term := Proof.Events729.exact186718RawTerms
def group : MergeGroup := .operator 186675 186718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186675) (leftOrdinal := 0)
    (rightResult := 186718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15812⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186722

namespace LeftMerge186739
def owner : Owner := ⟨.program ⟨257⟩, ⟨16322⟩⟩
def mergeEvent : Nat := 186739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events729.exact186736RawTerms
def group : MergeGroup := .relation 186738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 186738) (rhsResult := 186736)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 186737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩]⟩) (none) 186736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186739

namespace LeftMerge186740
def owner : Owner := ⟨.program ⟨257⟩, ⟨16322⟩⟩
def mergeEvent : Nat := 186740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩] } }
def rhsRaw : List Term := Proof.Events729.exact186736RawTerms
def group : MergeGroup := .relation 186738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 186738) (rhsResult := 186736)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 186737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩]⟩) (none) 186736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17392⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186740

namespace LeftMerge186741
def owner : Owner := ⟨.program ⟨257⟩, ⟨16322⟩⟩
def mergeEvent : Nat := 186741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16867⟩⟩] } }
def rhsRaw : List Term := Proof.Events729.exact186736RawTerms
def group : MergeGroup := .relation 186738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 186738) (rhsResult := 186736)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 186737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩]⟩) (none) 186736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16867⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], [⟨.program ⟨257⟩, ⟨16867⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge186741

namespace LeftMerge186742
def owner : Owner := ⟨.program ⟨257⟩, ⟨16322⟩⟩
def mergeEvent : Nat := 186742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events729.exact186736RawTerms
def group : MergeGroup := .relation 186738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 186738) (rhsResult := 186736)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 186737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16319⟩⟩]⟩) (none) 186736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15812⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨15812⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186742

namespace LeftMerge186747
def owner : Owner := ⟨.program ⟨257⟩, ⟨17394⟩⟩
def mergeEvent : Nat := 186747
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16867⟩⟩] } }
def leftRaw : List Term := Proof.Events729.exact186743RawTerms
def rightRaw : List Term := Proof.Events728.exact186557RawTerms
def group : MergeGroup := .operator 186743 186557
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 186743) (leftOrdinal := 2)
    (rightResult := 186557) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16867⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16867⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨12426⟩⟩, ⟨.program ⟨257⟩, ⟨15546⟩⟩], [⟨.program ⟨257⟩, ⟨16867⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge186747

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
