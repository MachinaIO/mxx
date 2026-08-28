import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge6940
def owner : Owner := ⟨.program ⟨214⟩, ⟨30208⟩⟩
def mergeEvent : Nat := 6940
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩] } }
def leftRaw : List Term := Proof.Events027.exact6935RawTerms
def rightRaw : List Term := Proof.Events026.exact6757RawTerms
def group : MergeGroup := .operator 6935 6757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6935) (leftOrdinal := 0)
    (rightResult := 6757) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6707⟩⟩, ⟨.program ⟨214⟩, ⟨30205⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6940

namespace LeftMerge6969
def owner : Owner := ⟨.program ⟨214⟩, ⟨13189⟩⟩
def mergeEvent : Nat := 6969
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact74RawTerms
def rightRaw : List Term := Proof.Events025.exact6449RawTerms
def group : MergeGroup := .operator 74 6449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74) (leftOrdinal := 0)
    (rightResult := 6449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6969

namespace LeftMerge6977
def owner : Owner := ⟨.program ⟨214⟩, ⟨7397⟩⟩
def mergeEvent : Nat := 6977
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events027.exact6973RawTerms
def group : MergeGroup := .operator 6314 6973
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 6973) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6977

namespace LeftMerge6994
def owner : Owner := ⟨.program ⟨214⟩, ⟨13192⟩⟩
def mergeEvent : Nat := 6994
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events027.exact6988RawTerms
def rightRaw : List Term := Proof.Events000.exact77RawTerms
def group : MergeGroup := .operator 6988 77
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6988) (leftOrdinal := 1)
    (rightResult := 77) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge6994

namespace LeftMerge6995
def owner : Owner := ⟨.program ⟨214⟩, ⟨13192⟩⟩
def mergeEvent : Nat := 6995
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }
def leftRaw : List Term := Proof.Events027.exact6988RawTerms
def rightRaw : List Term := Proof.Events000.exact77RawTerms
def group : MergeGroup := .operator 6988 77
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6988) (leftOrdinal := 0)
    (rightResult := 77) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge6995

namespace LeftMerge7010
def owner : Owner := ⟨.program ⟨214⟩, ⟨10261⟩⟩
def mergeEvent : Nat := 7010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events000.exact77RawTerms
def rightRaw : List Term := Proof.Events025.exact6449RawTerms
def group : MergeGroup := .operator 77 6449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77) (leftOrdinal := 0)
    (rightResult := 6449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7010

namespace LeftMerge7018
def owner : Owner := ⟨.program ⟨214⟩, ⟨7377⟩⟩
def mergeEvent : Nat := 7018
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6314RawTerms
def rightRaw : List Term := Proof.Events027.exact7014RawTerms
def group : MergeGroup := .operator 6314 7014
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6314) (leftOrdinal := 0)
    (rightResult := 7014) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7018

namespace LeftMerge7035
def owner : Owner := ⟨.program ⟨214⟩, ⟨10264⟩⟩
def mergeEvent : Nat := 7035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events027.exact7029RawTerms
def rightRaw : List Term := Proof.Events027.exact7003RawTerms
def group : MergeGroup := .operator 7029 7003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7029) (leftOrdinal := 1)
    (rightResult := 7003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7035

namespace LeftMerge7037
def owner : Owner := ⟨.program ⟨214⟩, ⟨10264⟩⟩
def mergeEvent : Nat := 7037
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact6973RawTerms
def group : MergeGroup := .relation 7036
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7036) (rhsResult := 6973)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7879⟩⟩) ⟨6789⟩ 6973) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7037

namespace LeftMerge7038
def owner : Owner := ⟨.program ⟨214⟩, ⟨10264⟩⟩
def mergeEvent : Nat := 7038
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events027.exact7029RawTerms
def rightRaw : List Term := Proof.Events027.exact7003RawTerms
def group : MergeGroup := .operator 7029 7003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7029) (leftOrdinal := 0)
    (rightResult := 7003) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7038

namespace LeftMerge7043
def owner : Owner := ⟨.program ⟨214⟩, ⟨13193⟩⟩
def mergeEvent : Nat := 7043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }
def leftRaw : List Term := Proof.Events027.exact7039RawTerms
def rightRaw : List Term := Proof.Events027.exact6996RawTerms
def group : MergeGroup := .operator 7039 6996
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7039) (leftOrdinal := 1)
    (rightResult := 6996) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6789⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩], [⟨.program ⟨214⟩, ⟨6789⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7043

namespace LeftMerge7051
def owner : Owner := ⟨.program ⟨214⟩, ⟨25702⟩⟩
def mergeEvent : Nat := 7051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩] } }
def leftRaw : List Term := Proof.Events027.exact7045RawTerms
def rightRaw : List Term := Proof.Events027.exact6962RawTerms
def group : MergeGroup := .operator 7045 6962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7045) (leftOrdinal := 1)
    (rightResult := 6962) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25701⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7051

namespace LeftMerge7053
def owner : Owner := ⟨.program ⟨214⟩, ⟨25702⟩⟩
def mergeEvent : Nat := 7053
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23382⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact6959RawTerms
def group : MergeGroup := .relation 7052
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7052) (rhsResult := 6959)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25701⟩⟩) ⟨23382⟩ 6959) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23382⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨23382⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7053

namespace LeftMerge7054
def owner : Owner := ⟨.program ⟨214⟩, ⟨25702⟩⟩
def mergeEvent : Nat := 7054
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩] } }
def leftRaw : List Term := Proof.Events027.exact7045RawTerms
def rightRaw : List Term := Proof.Events027.exact6962RawTerms
def group : MergeGroup := .operator 7045 6962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7045) (leftOrdinal := 0)
    (rightResult := 6962) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25701⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7054

namespace LeftMerge7068
def owner : Owner := ⟨.program ⟨214⟩, ⟨20195⟩⟩
def mergeEvent : Nat := 7068
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events027.exact7062RawTerms
def group : MergeGroup := .operator 6561 7062
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 7062) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20192⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7068

namespace LeftMerge7147
def owner : Owner := ⟨.program ⟨214⟩, ⟨13187⟩⟩
def mergeEvent : Nat := 7147
def frameStart : Nat := 7117
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events027.exact7143RawTerms
def rightRaw : List Term := Proof.Events027.exact7140RawTerms
def group : MergeGroup := .operator 7143 7140
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7143) (leftOrdinal := 0)
    (rightResult := 7140) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7147

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
