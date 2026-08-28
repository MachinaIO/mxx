import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge49934
def owner : Owner := ⟨.program ⟨214⟩, ⟨26586⟩⟩
def mergeEvent : Nat := 49934
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23789⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact49929RawTerms
def rightRaw : List Term := Proof.Events194.exact49751RawTerms
def group : MergeGroup := .operator 49929 49751
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49929) (leftOrdinal := 2)
    (rightResult := 49751) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23789⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23789⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14961⟩⟩], [⟨.program ⟨214⟩, ⟨23789⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49934

namespace LeftMerge49942
def owner : Owner := ⟨.program ⟨214⟩, ⟨26587⟩⟩
def mergeEvent : Nat := 49942
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact49936RawTerms
def rightRaw : List Term := Proof.Events022.exact5839RawTerms
def group : MergeGroup := .operator 49936 5839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49936) (leftOrdinal := 0)
    (rightResult := 5839) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6710⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6671⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6710⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49942

namespace LeftMerge49943
def owner : Owner := ⟨.program ⟨214⟩, ⟨26587⟩⟩
def mergeEvent : Nat := 49943
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact49936RawTerms
def rightRaw : List Term := Proof.Events022.exact5839RawTerms
def group : MergeGroup := .operator 49936 5839
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 49936) (leftOrdinal := 1)
    (rightResult := 5839) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6671⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49943

namespace LeftMerge49945
def owner : Owner := ⟨.program ⟨214⟩, ⟨26587⟩⟩
def mergeEvent : Nat := 49945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events022.exact5832RawTerms
def group : MergeGroup := .relation 49944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49944) (rhsResult := 5832)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6671⟩⟩) ⟨6607⟩ 5832) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6475⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6475⟩⟩, ⟨.program ⟨214⟩, ⟨15057⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49945

namespace LeftMerge49959
def owner : Owner := ⟨.program ⟨214⟩, ⟨26377⟩⟩
def mergeEvent : Nat := 49959
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44517RawTerms
def rightRaw : List Term := Proof.Events195.exact49953RawTerms
def group : MergeGroup := .operator 44517 49953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44517) (leftOrdinal := 0)
    (rightResult := 49953) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26375⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49959

namespace LeftMerge49960
def owner : Owner := ⟨.program ⟨214⟩, ⟨26377⟩⟩
def mergeEvent : Nat := 49960
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩] } }
def leftRaw : List Term := Proof.Events173.exact44517RawTerms
def rightRaw : List Term := Proof.Events195.exact49953RawTerms
def group : MergeGroup := .operator 44517 49953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 44517) (leftOrdinal := 1)
    (rightResult := 49953) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26375⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49960

namespace LeftMerge49962
def owner : Owner := ⟨.program ⟨214⟩, ⟨26377⟩⟩
def mergeEvent : Nat := 49962
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23726⟩⟩] } }
def rhsRaw : List Term := Proof.Events195.exact49950RawTerms
def group : MergeGroup := .relation 49961
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 49961) (rhsResult := 49950)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26375⟩⟩) ⟨23726⟩ 49950) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23726⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23726⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge49962

namespace LeftMerge49976
def owner : Owner := ⟨.program ⟨214⟩, ⟨20331⟩⟩
def mergeEvent : Nat := 49976
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events195.exact49970RawTerms
def group : MergeGroup := .operator 36137 49970
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 49970) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20328⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge49976

namespace LeftMerge50097
def owner : Owner := ⟨.program ⟨214⟩, ⟨14842⟩⟩
def mergeEvent : Nat := 50097
def frameStart : Nat := 50031
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50093RawTerms
def rightRaw : List Term := Proof.Events195.exact50091RawTerms
def group : MergeGroup := .operator 50093 50091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50093) (leftOrdinal := 0)
    (rightResult := 50091) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50097

namespace LeftMerge50109
def owner : Owner := ⟨.program ⟨214⟩, ⟨26376⟩⟩
def mergeEvent : Nat := 50109
def frameStart : Nat := 50031
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50105RawTerms
def rightRaw : List Term := Proof.Events195.exact50082RawTerms
def group : MergeGroup := .operator 50105 50082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50105) (leftOrdinal := 0)
    (rightResult := 50082) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26375⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50109

namespace LeftMerge50110
def owner : Owner := ⟨.program ⟨214⟩, ⟨26376⟩⟩
def mergeEvent : Nat := 50110
def frameStart : Nat := 50031
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50105RawTerms
def rightRaw : List Term := Proof.Events195.exact50082RawTerms
def group : MergeGroup := .operator 50105 50082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50105) (leftOrdinal := 1)
    (rightResult := 50082) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26375⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50110

namespace LeftMerge50112
def owner : Owner := ⟨.program ⟨214⟩, ⟨26376⟩⟩
def mergeEvent : Nat := 50112
def frameStart : Nat := 50031
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23726⟩⟩] } }
def rhsRaw : List Term := Proof.Events195.exact50079RawTerms
def group : MergeGroup := .relation 50111
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50111) (rhsResult := 50079)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26375⟩⟩) ⟨23726⟩ 50079) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23726⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23726⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50112

namespace LeftMerge50120
def owner : Owner := ⟨.program ⟨214⟩, ⟨14899⟩⟩
def mergeEvent : Nat := 50120
def frameStart : Nat := 50031
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events195.exact50093RawTerms
def rightRaw : List Term := Proof.Events195.exact50116RawTerms
def group : MergeGroup := .operator 50093 50116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50093) (leftOrdinal := 0)
    (rightResult := 50116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14896⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14896⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50120

namespace LeftMerge50137
def owner : Owner := ⟨.program ⟨214⟩, ⟨20331⟩⟩
def mergeEvent : Nat := 50137
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }
def rhsRaw : List Term := Proof.Events195.exact50134RawTerms
def group : MergeGroup := .relation 50136
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50136) (rhsResult := 50134)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50135 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩) (none) 50134) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6708⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6708⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50137

namespace LeftMerge50138
def owner : Owner := ⟨.program ⟨214⟩, ⟨20331⟩⟩
def mergeEvent : Nat := 50138
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩] } }
def rhsRaw : List Term := Proof.Events195.exact50134RawTerms
def group : MergeGroup := .relation 50136
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50136) (rhsResult := 50134)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50135 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩) (none) 50134) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6690⟩⟩, ⟨.program ⟨214⟩, ⟨26375⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge50138

namespace LeftMerge50139
def owner : Owner := ⟨.program ⟨214⟩, ⟨20331⟩⟩
def mergeEvent : Nat := 50139
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23726⟩⟩] } }
def rhsRaw : List Term := Proof.Events195.exact50134RawTerms
def group : MergeGroup := .relation 50136
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 50136) (rhsResult := 50134)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 50135 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20328⟩⟩]⟩) (none) 50134) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14800⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23726⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14800⟩⟩], [⟨.program ⟨214⟩, ⟨23726⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge50139

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
