import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge69712
def owner : Owner := ⟨.program ⟨257⟩, ⟨17439⟩⟩
def mergeEvent : Nat := 69712
def frameStart : Nat := 69619
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩] } }
def leftRaw : List Term := Proof.Events272.exact69707RawTerms
def rightRaw : List Term := Proof.Events272.exact69664RawTerms
def group : MergeGroup := .operator 69707 69664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69707) (leftOrdinal := 1)
    (rightResult := 69664) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17436⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69712

namespace LeftMerge69714
def owner : Owner := ⟨.program ⟨257⟩, ⟨17439⟩⟩
def mergeEvent : Nat := 69714
def frameStart : Nat := 69619
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16891⟩⟩] } }
def rhsRaw : List Term := Proof.Events272.exact69661RawTerms
def group : MergeGroup := .relation 69713
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69713) (rhsResult := 69661)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17436⟩⟩) ⟨16891⟩ 69661) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16891⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], [⟨.program ⟨257⟩, ⟨16891⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69714

namespace LeftMerge69722
def owner : Owner := ⟨.program ⟨257⟩, ⟨15846⟩⟩
def mergeEvent : Nat := 69722
def frameStart : Nat := 69619
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events272.exact69675RawTerms
def rightRaw : List Term := Proof.Events272.exact69718RawTerms
def group : MergeGroup := .operator 69675 69718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69675) (leftOrdinal := 0)
    (rightResult := 69718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69722

namespace LeftMerge69739
def owner : Owner := ⟨.program ⟨257⟩, ⟨16362⟩⟩
def mergeEvent : Nat := 69739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }
def rhsRaw : List Term := Proof.Events272.exact69736RawTerms
def group : MergeGroup := .relation 69738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69738) (rhsResult := 69736)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16359⟩⟩]⟩) (none) 69736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69739

namespace LeftMerge69740
def owner : Owner := ⟨.program ⟨257⟩, ⟨16362⟩⟩
def mergeEvent : Nat := 69740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩] } }
def rhsRaw : List Term := Proof.Events272.exact69736RawTerms
def group : MergeGroup := .relation 69738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69738) (rhsResult := 69736)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16359⟩⟩]⟩) (none) 69736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69740

namespace LeftMerge69741
def owner : Owner := ⟨.program ⟨257⟩, ⟨16362⟩⟩
def mergeEvent : Nat := 69741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16891⟩⟩] } }
def rhsRaw : List Term := Proof.Events272.exact69736RawTerms
def group : MergeGroup := .relation 69738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69738) (rhsResult := 69736)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16359⟩⟩]⟩) (none) 69736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16891⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], [⟨.program ⟨257⟩, ⟨16891⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69741

namespace LeftMerge69742
def owner : Owner := ⟨.program ⟨257⟩, ⟨16362⟩⟩
def mergeEvent : Nat := 69742
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events272.exact69736RawTerms
def group : MergeGroup := .relation 69738
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69738) (rhsResult := 69736)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16359⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69737 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16359⟩⟩]⟩) (none) 69736) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69742

namespace LeftMerge69747
def owner : Owner := ⟨.program ⟨257⟩, ⟨17438⟩⟩
def mergeEvent : Nat := 69747
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16891⟩⟩] } }
def leftRaw : List Term := Proof.Events272.exact69743RawTerms
def rightRaw : List Term := Proof.Events271.exact69557RawTerms
def group : MergeGroup := .operator 69743 69557
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69743) (leftOrdinal := 2)
    (rightResult := 69557) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16891⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16891⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12486⟩⟩, ⟨.program ⟨257⟩, ⟨15642⟩⟩], [⟨.program ⟨257⟩, ⟨16891⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69747

namespace LeftMerge69748
def owner : Owner := ⟨.program ⟨257⟩, ⟨17438⟩⟩
def mergeEvent : Nat := 69748
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩] } }
def leftRaw : List Term := Proof.Events272.exact69743RawTerms
def rightRaw : List Term := Proof.Events271.exact69557RawTerms
def group : MergeGroup := .operator 69743 69557
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69743) (leftOrdinal := 1)
    (rightResult := 69557) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17436⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69748

namespace LeftMerge69756
def owner : Owner := ⟨.program ⟨257⟩, ⟨17959⟩⟩
def mergeEvent : Nat := 69756
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩] } }
def leftRaw : List Term := Proof.Events272.exact69750RawTerms
def rightRaw : List Term := Proof.Events271.exact69473RawTerms
def group : MergeGroup := .operator 69750 69473
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69750) (leftOrdinal := 0)
    (rightResult := 69473) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17957⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69756

namespace LeftMerge69757
def owner : Owner := ⟨.program ⟨257⟩, ⟨17959⟩⟩
def mergeEvent : Nat := 69757
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩] } }
def leftRaw : List Term := Proof.Events272.exact69750RawTerms
def rightRaw : List Term := Proof.Events271.exact69473RawTerms
def group : MergeGroup := .operator 69750 69473
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69750) (leftOrdinal := 1)
    (rightResult := 69473) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17957⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69757

namespace LeftMerge69759
def owner : Owner := ⟨.program ⟨257⟩, ⟨17959⟩⟩
def mergeEvent : Nat := 69759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17064⟩⟩] } }
def rhsRaw : List Term := Proof.Events271.exact69470RawTerms
def group : MergeGroup := .relation 69758
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69758) (rhsResult := 69470)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17957⟩⟩) ⟨17064⟩ 69470) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17064⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨17064⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69759

namespace LeftMerge69773
def owner : Owner := ⟨.program ⟨257⟩, ⟨16739⟩⟩
def mergeEvent : Nat := 69773
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16736⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events272.exact69767RawTerms
def group : MergeGroup := .operator 61370 69767
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 69767) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16736⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16736⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69773

namespace LeftMerge69894
def owner : Owner := ⟨.program ⟨257⟩, ⟨17236⟩⟩
def mergeEvent : Nat := 69894
def frameStart : Nat := 69828
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact69890RawTerms
def rightRaw : List Term := Proof.Events273.exact69888RawTerms
def group : MergeGroup := .operator 69890 69888
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69890) (leftOrdinal := 0)
    (rightResult := 69888) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69894

namespace LeftMerge69906
def owner : Owner := ⟨.program ⟨257⟩, ⟨17958⟩⟩
def mergeEvent : Nat := 69906
def frameStart : Nat := 69828
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact69902RawTerms
def rightRaw : List Term := Proof.Events272.exact69879RawTerms
def group : MergeGroup := .operator 69902 69879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69902) (leftOrdinal := 0)
    (rightResult := 69879) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17957⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69906

namespace LeftMerge69907
def owner : Owner := ⟨.program ⟨257⟩, ⟨17958⟩⟩
def mergeEvent : Nat := 69907
def frameStart : Nat := 69828
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩] } }
def leftRaw : List Term := Proof.Events273.exact69902RawTerms
def rightRaw : List Term := Proof.Events272.exact69879RawTerms
def group : MergeGroup := .operator 69902 69879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69902) (leftOrdinal := 1)
    (rightResult := 69879) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17957⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17957⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69907

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
