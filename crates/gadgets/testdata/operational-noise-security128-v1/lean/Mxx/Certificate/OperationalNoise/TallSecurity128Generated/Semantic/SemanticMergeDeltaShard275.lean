import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge47884
def owner : Owner := ⟨.program ⟨257⟩, ⟨43312⟩⟩
def mergeEvent : Nat := 47884
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact47881RawTerms
def group : MergeGroup := .relation 47883
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 47883) (rhsResult := 47881)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 47882 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩) (none) 47881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge47884

namespace LeftMerge47885
def owner : Owner := ⟨.program ⟨257⟩, ⟨43312⟩⟩
def mergeEvent : Nat := 47885
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact47881RawTerms
def group : MergeGroup := .relation 47883
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 47883) (rhsResult := 47881)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 47882 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩) (none) 47881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge47885

namespace LeftMerge47886
def owner : Owner := ⟨.program ⟨257⟩, ⟨43312⟩⟩
def mergeEvent : Nat := 47886
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43837⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact47881RawTerms
def group : MergeGroup := .relation 47883
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 47883) (rhsResult := 47881)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 47882 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩) (none) 47881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43837⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge47886

namespace LeftMerge47887
def owner : Owner := ⟨.program ⟨257⟩, ⟨43312⟩⟩
def mergeEvent : Nat := 47887
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact47881RawTerms
def group : MergeGroup := .relation 47883
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 47883) (rhsResult := 47881)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 47882 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43309⟩⟩]⟩) (none) 47881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge47887

namespace LeftMerge47892
def owner : Owner := ⟨.program ⟨257⟩, ⟨44389⟩⟩
def mergeEvent : Nat := 47892
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43837⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact47888RawTerms
def rightRaw : List Term := Proof.Events186.exact47702RawTerms
def group : MergeGroup := .operator 47888 47702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 47888) (leftOrdinal := 2)
    (rightResult := 47702) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43837⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨43837⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨14601⟩⟩, ⟨.program ⟨257⟩, ⟨42666⟩⟩], [⟨.program ⟨257⟩, ⟨43837⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge47892

namespace LeftMerge47893
def owner : Owner := ⟨.program ⟨257⟩, ⟨44389⟩⟩
def mergeEvent : Nat := 47893
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact47888RawTerms
def rightRaw : List Term := Proof.Events186.exact47702RawTerms
def group : MergeGroup := .operator 47888 47702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 47888) (leftOrdinal := 1)
    (rightResult := 47702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩, ⟨.program ⟨257⟩, ⟨44387⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge47893

namespace LeftMerge47901
def owner : Owner := ⟨.program ⟨257⟩, ⟨44871⟩⟩
def mergeEvent : Nat := 47901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact47895RawTerms
def rightRaw : List Term := Proof.Events186.exact47618RawTerms
def group : MergeGroup := .operator 47895 47618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 47895) (leftOrdinal := 0)
    (rightResult := 47618) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge47901

namespace LeftMerge47902
def owner : Owner := ⟨.program ⟨257⟩, ⟨44871⟩⟩
def mergeEvent : Nat := 47902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact47895RawTerms
def rightRaw : List Term := Proof.Events186.exact47618RawTerms
def group : MergeGroup := .operator 47895 47618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 47895) (leftOrdinal := 1)
    (rightResult := 47618) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge47902

namespace LeftMerge47904
def owner : Owner := ⟨.program ⟨257⟩, ⟨44871⟩⟩
def mergeEvent : Nat := 47904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨44013⟩⟩] } }
def rhsRaw : List Term := Proof.Events185.exact47615RawTerms
def group : MergeGroup := .relation 47903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 47903) (rhsResult := 47615)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44869⟩⟩) ⟨44013⟩ 47615) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44013⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge47904

namespace LeftMerge47918
def owner : Owner := ⟨.program ⟨257⟩, ⟨43699⟩⟩
def mergeEvent : Nat := 47918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events187.exact47912RawTerms
def group : MergeGroup := .operator 46745 47912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 47912) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨43696⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge47918

namespace LeftMerge48039
def owner : Owner := ⟨.program ⟨257⟩, ⟨44180⟩⟩
def mergeEvent : Nat := 48039
def frameStart : Nat := 47973
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact48035RawTerms
def rightRaw : List Term := Proof.Events187.exact48033RawTerms
def group : MergeGroup := .operator 48035 48033
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48035) (leftOrdinal := 0)
    (rightResult := 48033) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48039

namespace LeftMerge48051
def owner : Owner := ⟨.program ⟨257⟩, ⟨44870⟩⟩
def mergeEvent : Nat := 48051
def frameStart : Nat := 47973
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact48047RawTerms
def rightRaw : List Term := Proof.Events187.exact48024RawTerms
def group : MergeGroup := .operator 48047 48024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48047) (leftOrdinal := 0)
    (rightResult := 48024) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7194⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44869⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7194⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48051

namespace LeftMerge48052
def owner : Owner := ⟨.program ⟨257⟩, ⟨44870⟩⟩
def mergeEvent : Nat := 48052
def frameStart : Nat := 47973
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact48047RawTerms
def rightRaw : List Term := Proof.Events187.exact48024RawTerms
def group : MergeGroup := .operator 48047 48024
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48047) (leftOrdinal := 1)
    (rightResult := 48024) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48052

namespace LeftMerge48054
def owner : Owner := ⟨.program ⟨257⟩, ⟨44870⟩⟩
def mergeEvent : Nat := 48054
def frameStart : Nat := 47973
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨44013⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact48021RawTerms
def group : MergeGroup := .relation 48053
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48053) (rhsResult := 48021)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨44869⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨44869⟩⟩) ⟨44013⟩ 48021) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨44013⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42852⟩⟩], [⟨.program ⟨257⟩, ⟨44013⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48054

namespace LeftMerge48062
def owner : Owner := ⟨.program ⟨257⟩, ⟨43104⟩⟩
def mergeEvent : Nat := 48062
def frameStart : Nat := 47973
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact48035RawTerms
def rightRaw : List Term := Proof.Events187.exact48058RawTerms
def group : MergeGroup := .operator 48035 48058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48035) (leftOrdinal := 0)
    (rightResult := 48058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43103⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48062

namespace LeftMerge48079
def owner : Owner := ⟨.program ⟨257⟩, ⟨43699⟩⟩
def mergeEvent : Nat := 48079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact48076RawTerms
def group : MergeGroup := .relation 48078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48078) (rhsResult := 48076)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨43696⟩⟩]⟩) (none) 48076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48079

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
