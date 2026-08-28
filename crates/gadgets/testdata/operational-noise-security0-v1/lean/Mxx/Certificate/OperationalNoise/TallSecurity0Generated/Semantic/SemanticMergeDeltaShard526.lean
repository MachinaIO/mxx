import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge85908
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def mergeEvent : Nat := 85908
def frameStart : Nat := 85827
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact85904RawTerms
def rightRaw : List Term := Proof.Events335.exact85901RawTerms
def group : MergeGroup := .operator 85904 85901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85904) (leftOrdinal := 0)
    (rightResult := 85901) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7846⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85908

namespace LeftMerge85917
def owner : Owner := ⟨.program ⟨214⟩, ⟨25915⟩⟩
def mergeEvent : Nat := 85917
def frameStart : Nat := 85827
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact85913RawTerms
def rightRaw : List Term := Proof.Events335.exact85872RawTerms
def group : MergeGroup := .operator 85913 85872
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85913) (leftOrdinal := 0)
    (rightResult := 85872) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25912⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85917

namespace LeftMerge85918
def owner : Owner := ⟨.program ⟨214⟩, ⟨25915⟩⟩
def mergeEvent : Nat := 85918
def frameStart : Nat := 85827
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact85913RawTerms
def rightRaw : List Term := Proof.Events335.exact85872RawTerms
def group : MergeGroup := .operator 85913 85872
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85913) (leftOrdinal := 1)
    (rightResult := 85872) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25912⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85918

namespace LeftMerge85920
def owner : Owner := ⟨.program ⟨214⟩, ⟨25915⟩⟩
def mergeEvent : Nat := 85920
def frameStart : Nat := 85827
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23500⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85869RawTerms
def group : MergeGroup := .relation 85919
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85919) (rhsResult := 85869)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25912⟩⟩) ⟨23500⟩ 85869) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23500⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85920

namespace LeftMerge85928
def owner : Owner := ⟨.program ⟨214⟩, ⟨15704⟩⟩
def mergeEvent : Nat := 85928
def frameStart : Nat := 85827
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15702⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact85883RawTerms
def rightRaw : List Term := Proof.Events335.exact85924RawTerms
def group : MergeGroup := .operator 85883 85924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85883) (leftOrdinal := 0)
    (rightResult := 85924) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15702⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85928

namespace LeftMerge85945
def owner : Owner := ⟨.program ⟨214⟩, ⟨19387⟩⟩
def mergeEvent : Nat := 85945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85942RawTerms
def group : MergeGroup := .relation 85944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85944) (rhsResult := 85942)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85943 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩) (none) 85942) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85945

namespace LeftMerge85946
def owner : Owner := ⟨.program ⟨214⟩, ⟨19387⟩⟩
def mergeEvent : Nat := 85946
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85942RawTerms
def group : MergeGroup := .relation 85944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85944) (rhsResult := 85942)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85943 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩) (none) 85942) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85946

namespace LeftMerge85947
def owner : Owner := ⟨.program ⟨214⟩, ⟨19387⟩⟩
def mergeEvent : Nat := 85947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23500⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85942RawTerms
def group : MergeGroup := .relation 85944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85944) (rhsResult := 85942)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85943 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩) (none) 85942) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23500⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85947

namespace LeftMerge85948
def owner : Owner := ⟨.program ⟨214⟩, ⟨19387⟩⟩
def mergeEvent : Nat := 85948
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events335.exact85942RawTerms
def group : MergeGroup := .relation 85944
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85944) (rhsResult := 85942)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 85943 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19384⟩⟩]⟩) (none) 85942) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15702⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85948

namespace LeftMerge85953
def owner : Owner := ⟨.program ⟨214⟩, ⟨25914⟩⟩
def mergeEvent : Nat := 85953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23500⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact85949RawTerms
def rightRaw : List Term := Proof.Events335.exact85765RawTerms
def group : MergeGroup := .operator 85949 85765
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85949) (leftOrdinal := 2)
    (rightResult := 85765) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23500⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23500⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨11301⟩⟩, ⟨.program ⟨214⟩, ⟨13773⟩⟩], [⟨.program ⟨214⟩, ⟨23500⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85953

namespace LeftMerge85954
def owner : Owner := ⟨.program ⟨214⟩, ⟨25914⟩⟩
def mergeEvent : Nat := 85954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact85949RawTerms
def rightRaw : List Term := Proof.Events335.exact85765RawTerms
def group : MergeGroup := .operator 85949 85765
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85949) (leftOrdinal := 1)
    (rightResult := 85765) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6794⟩⟩, ⟨.program ⟨214⟩, ⟨7846⟩⟩, ⟨.program ⟨214⟩, ⟨25912⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85954

namespace LeftMerge85962
def owner : Owner := ⟨.program ⟨214⟩, ⟨27434⟩⟩
def mergeEvent : Nat := 85962
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact85956RawTerms
def rightRaw : List Term := Proof.Events334.exact85681RawTerms
def group : MergeGroup := .operator 85956 85681
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85956) (leftOrdinal := 0)
    (rightResult := 85681) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6695⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27432⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6695⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85962

namespace LeftMerge85963
def owner : Owner := ⟨.program ⟨214⟩, ⟨27434⟩⟩
def mergeEvent : Nat := 85963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩] } }
def leftRaw : List Term := Proof.Events335.exact85956RawTerms
def rightRaw : List Term := Proof.Events334.exact85681RawTerms
def group : MergeGroup := .operator 85956 85681
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 85956) (leftOrdinal := 1)
    (rightResult := 85681) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨27432⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85963

namespace LeftMerge85965
def owner : Owner := ⟨.program ⟨214⟩, ⟨27434⟩⟩
def mergeEvent : Nat := 85965
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24036⟩⟩] } }
def rhsRaw : List Term := Proof.Events334.exact85678RawTerms
def group : MergeGroup := .relation 85964
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 85964) (rhsResult := 85678)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨27432⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨27432⟩⟩) ⟨24036⟩ 85678) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24036⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨24036⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge85965

namespace LeftMerge85979
def owner : Owner := ⟨.program ⟨214⟩, ⟨21115⟩⟩
def mergeEvent : Nat := 85979
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21112⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events335.exact85973RawTerms
def group : MergeGroup := .operator 80012 85973
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 85973) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21112⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21112⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge85979

namespace LeftMerge86100
def owner : Owner := ⟨.program ⟨214⟩, ⟨15779⟩⟩
def mergeEvent : Nat := 86100
def frameStart : Nat := 86034
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15702⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events336.exact86096RawTerms
def rightRaw : List Term := Proof.Events336.exact86094RawTerms
def group : MergeGroup := .operator 86096 86094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 86096) (leftOrdinal := 0)
    (rightResult := 86094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15702⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15702⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge86100

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
