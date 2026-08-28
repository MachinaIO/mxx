import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge7177
def owner : Owner := ⟨.program ⟨214⟩, ⟨13268⟩⟩
def mergeEvent : Nat := 7177
def frameStart : Nat := 7117
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events028.exact7173RawTerms
def rightRaw : List Term := Proof.Events028.exact7171RawTerms
def group : MergeGroup := .operator 7173 7171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7173) (leftOrdinal := 0)
    (rightResult := 7171) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7177

namespace LeftMerge7200
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def mergeEvent : Nat := 7200
def frameStart : Nat := 7117
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }
def leftRaw : List Term := Proof.Events028.exact7196RawTerms
def rightRaw : List Term := Proof.Events028.exact7193RawTerms
def group : MergeGroup := .operator 7196 7193
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7196) (leftOrdinal := 0)
    (rightResult := 7193) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7879⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7200

namespace LeftMerge7209
def owner : Owner := ⟨.program ⟨214⟩, ⟨25704⟩⟩
def mergeEvent : Nat := 7209
def frameStart : Nat := 7117
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩] } }
def leftRaw : List Term := Proof.Events028.exact7205RawTerms
def rightRaw : List Term := Proof.Events027.exact7162RawTerms
def group : MergeGroup := .operator 7205 7162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7205) (leftOrdinal := 1)
    (rightResult := 7162) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25701⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7209

namespace LeftMerge7211
def owner : Owner := ⟨.program ⟨214⟩, ⟨25704⟩⟩
def mergeEvent : Nat := 7211
def frameStart : Nat := 7117
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23382⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact7159RawTerms
def group : MergeGroup := .relation 7210
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7210) (rhsResult := 7159)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25701⟩⟩) ⟨23382⟩ 7159) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23382⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨23382⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7211

namespace LeftMerge7212
def owner : Owner := ⟨.program ⟨214⟩, ⟨25704⟩⟩
def mergeEvent : Nat := 7212
def frameStart : Nat := 7117
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩] } }
def leftRaw : List Term := Proof.Events028.exact7205RawTerms
def rightRaw : List Term := Proof.Events027.exact7162RawTerms
def group : MergeGroup := .operator 7205 7162
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7205) (leftOrdinal := 0)
    (rightResult := 7162) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25701⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7212

namespace LeftMerge7220
def owner : Owner := ⟨.program ⟨214⟩, ⟨16889⟩⟩
def mergeEvent : Nat := 7220
def frameStart : Nat := 7117
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16887⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events028.exact7173RawTerms
def rightRaw : List Term := Proof.Events028.exact7216RawTerms
def group : MergeGroup := .operator 7173 7216
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7173) (leftOrdinal := 0)
    (rightResult := 7216) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16887⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7220

namespace LeftMerge7237
def owner : Owner := ⟨.program ⟨214⟩, ⟨20195⟩⟩
def mergeEvent : Nat := 7237
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23382⟩⟩] } }
def rhsRaw : List Term := Proof.Events028.exact7234RawTerms
def group : MergeGroup := .relation 7236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7236) (rhsResult := 7234)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 7235 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩) (none) 7234) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23382⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨23382⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7237

namespace LeftMerge7238
def owner : Owner := ⟨.program ⟨214⟩, ⟨20195⟩⟩
def mergeEvent : Nat := 7238
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩] } }
def rhsRaw : List Term := Proof.Events028.exact7234RawTerms
def group : MergeGroup := .relation 7236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7236) (rhsResult := 7234)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 7235 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩) (none) 7234) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7238

namespace LeftMerge7239
def owner : Owner := ⟨.program ⟨214⟩, ⟨20195⟩⟩
def mergeEvent : Nat := 7239
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events028.exact7234RawTerms
def group : MergeGroup := .relation 7236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7236) (rhsResult := 7234)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 7235 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩) (none) 7234) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16887⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7239

namespace LeftMerge7240
def owner : Owner := ⟨.program ⟨214⟩, ⟨20195⟩⟩
def mergeEvent : Nat := 7240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }
def rhsRaw : List Term := Proof.Events028.exact7234RawTerms
def group : MergeGroup := .relation 7236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7236) (rhsResult := 7234)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 7235 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20192⟩⟩]⟩) (none) 7234) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7240

namespace LeftMerge7245
def owner : Owner := ⟨.program ⟨214⟩, ⟨25703⟩⟩
def mergeEvent : Nat := 7245
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23382⟩⟩] } }
def leftRaw : List Term := Proof.Events028.exact7241RawTerms
def rightRaw : List Term := Proof.Events027.exact7055RawTerms
def group : MergeGroup := .operator 7241 7055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7241) (leftOrdinal := 2)
    (rightResult := 7055) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23382⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23382⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨10260⟩⟩, ⟨.program ⟨214⟩, ⟨13186⟩⟩], [⟨.program ⟨214⟩, ⟨23382⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7245

namespace LeftMerge7246
def owner : Owner := ⟨.program ⟨214⟩, ⟨25703⟩⟩
def mergeEvent : Nat := 7246
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩] } }
def leftRaw : List Term := Proof.Events028.exact7241RawTerms
def rightRaw : List Term := Proof.Events027.exact7055RawTerms
def group : MergeGroup := .operator 7241 7055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7241) (leftOrdinal := 1)
    (rightResult := 7055) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6769⟩⟩, ⟨.program ⟨214⟩, ⟨7879⟩⟩, ⟨.program ⟨214⟩, ⟨25701⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7246

namespace LeftMerge7254
def owner : Owner := ⟨.program ⟨214⟩, ⟨29873⟩⟩
def mergeEvent : Nat := 7254
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29871⟩⟩] } }
def leftRaw : List Term := Proof.Events028.exact7248RawTerms
def rightRaw : List Term := Proof.Events027.exact6952RawTerms
def group : MergeGroup := .operator 7248 6952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7248) (leftOrdinal := 1)
    (rightResult := 6952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29871⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29871⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7254

namespace LeftMerge7256
def owner : Owner := ⟨.program ⟨214⟩, ⟨29873⟩⟩
def mergeEvent : Nat := 7256
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24741⟩⟩] } }
def rhsRaw : List Term := Proof.Events027.exact6949RawTerms
def group : MergeGroup := .relation 7255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 7255) (rhsResult := 6949)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29871⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29871⟩⟩) ⟨24741⟩ 6949) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24741⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨16887⟩⟩], [⟨.program ⟨214⟩, ⟨24741⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge7256

namespace LeftMerge7257
def owner : Owner := ⟨.program ⟨214⟩, ⟨29873⟩⟩
def mergeEvent : Nat := 7257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29871⟩⟩] } }
def leftRaw : List Term := Proof.Events028.exact7248RawTerms
def rightRaw : List Term := Proof.Events027.exact6952RawTerms
def group : MergeGroup := .operator 7248 6952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7248) (leftOrdinal := 0)
    (rightResult := 6952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29871⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29871⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7257

namespace LeftMerge7271
def owner : Owner := ⟨.program ⟨214⟩, ⟨22715⟩⟩
def mergeEvent : Nat := 7271
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22712⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events028.exact7265RawTerms
def group : MergeGroup := .operator 6561 7265
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 7265) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22712⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22712⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge7271

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
