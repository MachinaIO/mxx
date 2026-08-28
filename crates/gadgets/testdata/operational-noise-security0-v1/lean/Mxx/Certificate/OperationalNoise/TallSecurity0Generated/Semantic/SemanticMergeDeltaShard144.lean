import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge24746
def owner : Owner := ⟨.program ⟨214⟩, ⟨28774⟩⟩
def mergeEvent : Nat := 24746
def frameStart : Nat := 24668
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24742RawTerms
def rightRaw : List Term := Proof.Events096.exact24719RawTerms
def group : MergeGroup := .operator 24742 24719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24742) (leftOrdinal := 0)
    (rightResult := 24719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28773⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24746

namespace LeftMerge24747
def owner : Owner := ⟨.program ⟨214⟩, ⟨28774⟩⟩
def mergeEvent : Nat := 24747
def frameStart : Nat := 24668
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16393⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24742RawTerms
def rightRaw : List Term := Proof.Events096.exact24719RawTerms
def group : MergeGroup := .operator 24742 24719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24742) (leftOrdinal := 1)
    (rightResult := 24719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16393⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28773⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24747

namespace LeftMerge24749
def owner : Owner := ⟨.program ⟨214⟩, ⟨28774⟩⟩
def mergeEvent : Nat := 24749
def frameStart : Nat := 24668
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16393⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24423⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24716RawTerms
def group : MergeGroup := .relation 24748
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24748) (rhsResult := 24716)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28773⟩⟩) ⟨24423⟩ 24716) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24423⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24749

namespace LeftMerge24757
def owner : Owner := ⟨.program ⟨214⟩, ⟨17130⟩⟩
def mergeEvent : Nat := 24757
def frameStart : Nat := 24668
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17129⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24730RawTerms
def rightRaw : List Term := Proof.Events096.exact24753RawTerms
def group : MergeGroup := .operator 24730 24753
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24730) (leftOrdinal := 0)
    (rightResult := 24753) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17129⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24757

namespace LeftMerge24774
def owner : Owner := ⟨.program ⟨214⟩, ⟨21991⟩⟩
def mergeEvent : Nat := 24774
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24771RawTerms
def group : MergeGroup := .relation 24773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24773) (rhsResult := 24771)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24772 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩) (none) 24771) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6731⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24774

namespace LeftMerge24775
def owner : Owner := ⟨.program ⟨214⟩, ⟨21991⟩⟩
def mergeEvent : Nat := 24775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24771RawTerms
def group : MergeGroup := .relation 24773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24773) (rhsResult := 24771)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24772 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩) (none) 24771) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24775

namespace LeftMerge24776
def owner : Owner := ⟨.program ⟨214⟩, ⟨21991⟩⟩
def mergeEvent : Nat := 24776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24423⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24771RawTerms
def group : MergeGroup := .relation 24773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24773) (rhsResult := 24771)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24772 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩) (none) 24771) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16393⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24423⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24776

namespace LeftMerge24777
def owner : Owner := ⟨.program ⟨214⟩, ⟨21991⟩⟩
def mergeEvent : Nat := 24777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events096.exact24771RawTerms
def group : MergeGroup := .relation 24773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 24773) (rhsResult := 24771)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 24772 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21988⟩⟩]⟩) (none) 24771) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17129⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨17129⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24777

namespace LeftMerge24782
def owner : Owner := ⟨.program ⟨214⟩, ⟨28776⟩⟩
def mergeEvent : Nat := 24782
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24778RawTerms
def rightRaw : List Term := Proof.Events096.exact24600RawTerms
def group : MergeGroup := .operator 24778 24600
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24778) (leftOrdinal := 0)
    (rightResult := 24600) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24782

namespace LeftMerge24783
def owner : Owner := ⟨.program ⟨214⟩, ⟨28776⟩⟩
def mergeEvent : Nat := 24783
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24423⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24778RawTerms
def rightRaw : List Term := Proof.Events096.exact24600RawTerms
def group : MergeGroup := .operator 24778 24600
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24778) (leftOrdinal := 2)
    (rightResult := 24600) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24423⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24423⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16393⟩⟩], [⟨.program ⟨214⟩, ⟨24423⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24783

namespace LeftMerge24809
def owner : Owner := ⟨.program ⟨214⟩, ⟨11788⟩⟩
def mergeEvent : Nat := 24809
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events003.exact1003RawTerms
def rightRaw : List Term := Proof.Events083.exact21420RawTerms
def group : MergeGroup := .operator 1003 21420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1003) (leftOrdinal := 0)
    (rightResult := 21420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24809

namespace LeftMerge24814
def owner : Owner := ⟨.program ⟨214⟩, ⟨7353⟩⟩
def mergeEvent : Nat := 24814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21290RawTerms
def rightRaw : List Term := Proof.Events038.exact9979RawTerms
def group : MergeGroup := .operator 21290 9979
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21290) (leftOrdinal := 0)
    (rightResult := 9979) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24814

namespace LeftMerge24831
def owner : Owner := ⟨.program ⟨214⟩, ⟨11791⟩⟩
def mergeEvent : Nat := 24831
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24825RawTerms
def rightRaw : List Term := Proof.Events003.exact1006RawTerms
def group : MergeGroup := .operator 24825 1006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24825) (leftOrdinal := 1)
    (rightResult := 1006) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩, ⟨.program ⟨214⟩, ⟨11785⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge24831

namespace LeftMerge24832
def owner : Owner := ⟨.program ⟨214⟩, ⟨11791⟩⟩
def mergeEvent : Nat := 24832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }
def leftRaw : List Term := Proof.Events096.exact24825RawTerms
def rightRaw : List Term := Proof.Events003.exact1006RawTerms
def group : MergeGroup := .operator 24825 1006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 24825) (leftOrdinal := 0)
    (rightResult := 1006) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6783⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], [⟨.program ⟨214⟩, ⟨6783⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24832

namespace LeftMerge24837
def owner : Owner := ⟨.program ⟨214⟩, ⟨9626⟩⟩
def mergeEvent : Nat := 24837
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events003.exact1006RawTerms
def rightRaw : List Term := Proof.Events083.exact21420RawTerms
def group : MergeGroup := .operator 1006 21420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1006) (leftOrdinal := 0)
    (rightResult := 21420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9625⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9625⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24837

namespace LeftMerge24842
def owner : Owner := ⟨.program ⟨214⟩, ⟨7333⟩⟩
def mergeEvent : Nat := 24842
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21290RawTerms
def rightRaw : List Term := Proof.Events039.exact10020RawTerms
def group : MergeGroup := .operator 21290 10020
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21290) (leftOrdinal := 0)
    (rightResult := 10020) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6763⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6763⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge24842

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
