import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge43035
def owner : Owner := ⟨.program ⟨214⟩, ⟨25309⟩⟩
def mergeEvent : Nat := 43035
def frameStart : Nat := 42940
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23168⟩⟩] } }
def rhsRaw : List Term := Proof.Events167.exact42982RawTerms
def group : MergeGroup := .relation 43034
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43034) (rhsResult := 42982)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25306⟩⟩) ⟨23168⟩ 42982) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23168⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨23168⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43035

namespace LeftMerge43043
def owner : Owner := ⟨.program ⟨214⟩, ⟨15432⟩⟩
def mergeEvent : Nat := 43043
def frameStart : Nat := 42940
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events167.exact42996RawTerms
def rightRaw : List Term := Proof.Events168.exact43039RawTerms
def group : MergeGroup := .operator 42996 43039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 42996) (leftOrdinal := 0)
    (rightResult := 43039) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43043

namespace LeftMerge43060
def owner : Owner := ⟨.program ⟨214⟩, ⟨19251⟩⟩
def mergeEvent : Nat := 43060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }
def rhsRaw : List Term := Proof.Events168.exact43057RawTerms
def group : MergeGroup := .relation 43059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43059) (rhsResult := 43057)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43058 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩]⟩) (none) 43057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43060

namespace LeftMerge43061
def owner : Owner := ⟨.program ⟨214⟩, ⟨19251⟩⟩
def mergeEvent : Nat := 43061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩] } }
def rhsRaw : List Term := Proof.Events168.exact43057RawTerms
def group : MergeGroup := .relation 43059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43059) (rhsResult := 43057)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43058 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩]⟩) (none) 43057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43061

namespace LeftMerge43062
def owner : Owner := ⟨.program ⟨214⟩, ⟨19251⟩⟩
def mergeEvent : Nat := 43062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23168⟩⟩] } }
def rhsRaw : List Term := Proof.Events168.exact43057RawTerms
def group : MergeGroup := .relation 43059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43059) (rhsResult := 43057)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43058 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩]⟩) (none) 43057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23168⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨23168⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43062

namespace LeftMerge43063
def owner : Owner := ⟨.program ⟨214⟩, ⟨19251⟩⟩
def mergeEvent : Nat := 43063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events168.exact43057RawTerms
def group : MergeGroup := .relation 43059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43059) (rhsResult := 43057)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43058 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19248⟩⟩]⟩) (none) 43057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43063

namespace LeftMerge43068
def owner : Owner := ⟨.program ⟨214⟩, ⟨25308⟩⟩
def mergeEvent : Nat := 43068
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23168⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43064RawTerms
def rightRaw : List Term := Proof.Events167.exact42878RawTerms
def group : MergeGroup := .operator 43064 42878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43064) (leftOrdinal := 2)
    (rightResult := 42878) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23168⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23168⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11141⟩⟩, ⟨.program ⟨214⟩, ⟨12181⟩⟩], [⟨.program ⟨214⟩, ⟨23168⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43068

namespace LeftMerge43069
def owner : Owner := ⟨.program ⟨214⟩, ⟨25308⟩⟩
def mergeEvent : Nat := 43069
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43064RawTerms
def rightRaw : List Term := Proof.Events167.exact42878RawTerms
def group : MergeGroup := .operator 43064 42878
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43064) (leftOrdinal := 1)
    (rightResult := 42878) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43069

namespace LeftMerge43077
def owner : Owner := ⟨.program ⟨214⟩, ⟨27026⟩⟩
def mergeEvent : Nat := 43077
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43071RawTerms
def rightRaw : List Term := Proof.Events167.exact42794RawTerms
def group : MergeGroup := .operator 43071 42794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43071) (leftOrdinal := 0)
    (rightResult := 42794) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27024⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43077

namespace LeftMerge43078
def owner : Owner := ⟨.program ⟨214⟩, ⟨27026⟩⟩
def mergeEvent : Nat := 43078
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43071RawTerms
def rightRaw : List Term := Proof.Events167.exact42794RawTerms
def group : MergeGroup := .operator 43071 42794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43071) (leftOrdinal := 1)
    (rightResult := 42794) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27024⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43078

namespace LeftMerge43080
def owner : Owner := ⟨.program ⟨214⟩, ⟨27026⟩⟩
def mergeEvent : Nat := 43080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23916⟩⟩] } }
def rhsRaw : List Term := Proof.Events167.exact42791RawTerms
def group : MergeGroup := .relation 43079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43079) (rhsResult := 42791)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27024⟩⟩) ⟨23916⟩ 42791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23916⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨23916⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43080

namespace LeftMerge43094
def owner : Owner := ⟨.program ⟨214⟩, ⟨20835⟩⟩
def mergeEvent : Nat := 43094
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20832⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events168.exact43088RawTerms
def group : MergeGroup := .operator 36137 43088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 43088) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20832⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20832⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43094

namespace LeftMerge43215
def owner : Owner := ⟨.program ⟨214⟩, ⟨15472⟩⟩
def mergeEvent : Nat := 43215
def frameStart : Nat := 43149
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43211RawTerms
def rightRaw : List Term := Proof.Events168.exact43209RawTerms
def group : MergeGroup := .operator 43211 43209
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43211) (leftOrdinal := 0)
    (rightResult := 43209) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43215

namespace LeftMerge43227
def owner : Owner := ⟨.program ⟨214⟩, ⟨27025⟩⟩
def mergeEvent : Nat := 43227
def frameStart : Nat := 43149
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43223RawTerms
def rightRaw : List Term := Proof.Events168.exact43200RawTerms
def group : MergeGroup := .operator 43223 43200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43223) (leftOrdinal := 0)
    (rightResult := 43200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27024⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6693⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43227

namespace LeftMerge43228
def owner : Owner := ⟨.program ⟨214⟩, ⟨27025⟩⟩
def mergeEvent : Nat := 43228
def frameStart : Nat := 43149
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43223RawTerms
def rightRaw : List Term := Proof.Events168.exact43200RawTerms
def group : MergeGroup := .operator 43223 43200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43223) (leftOrdinal := 1)
    (rightResult := 43200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27024⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43228

namespace LeftMerge43230
def owner : Owner := ⟨.program ⟨214⟩, ⟨27025⟩⟩
def mergeEvent : Nat := 43230
def frameStart : Nat := 43149
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15430⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23916⟩⟩] } }
def rhsRaw : List Term := Proof.Events168.exact43197RawTerms
def group : MergeGroup := .relation 43229
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43229) (rhsResult := 43197)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27024⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27024⟩⟩) ⟨23916⟩ 43197) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23916⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15430⟩⟩], [⟨.program ⟨214⟩, ⟨23916⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43230

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
