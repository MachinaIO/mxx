import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge13049
def owner : Owner := ⟨.program ⟨214⟩, ⟨13599⟩⟩
def mergeEvent : Nat := 13049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }
def rhsRaw : List Term := Proof.Events050.exact12985RawTerms
def group : MergeGroup := .relation 13048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13048) (rhsResult := 12985)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7843⟩⟩) ⟨6776⟩ 12985) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13049

namespace LeftMerge13050
def owner : Owner := ⟨.program ⟨214⟩, ⟨13599⟩⟩
def mergeEvent : Nat := 13050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }
def leftRaw : List Term := Proof.Events050.exact13041RawTerms
def rightRaw : List Term := Proof.Events050.exact13015RawTerms
def group : MergeGroup := .operator 13041 13015
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13041) (leftOrdinal := 0)
    (rightResult := 13015) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7843⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13050

namespace LeftMerge13055
def owner : Owner := ⟨.program ⟨214⟩, ⟨13600⟩⟩
def mergeEvent : Nat := 13055
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }
def leftRaw : List Term := Proof.Events050.exact13051RawTerms
def rightRaw : List Term := Proof.Events050.exact13008RawTerms
def group : MergeGroup := .operator 13051 13008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13051) (leftOrdinal := 1)
    (rightResult := 13008) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6776⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨6776⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13055

namespace LeftMerge13063
def owner : Owner := ⟨.program ⟨214⟩, ⟨25856⟩⟩
def mergeEvent : Nat := 13063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩] } }
def leftRaw : List Term := Proof.Events051.exact13057RawTerms
def rightRaw : List Term := Proof.Events050.exact12974RawTerms
def group : MergeGroup := .operator 13057 12974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13057) (leftOrdinal := 1)
    (rightResult := 12974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25855⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13063

namespace LeftMerge13065
def owner : Owner := ⟨.program ⟨214⟩, ⟨25856⟩⟩
def mergeEvent : Nat := 13065
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23466⟩⟩] } }
def rhsRaw : List Term := Proof.Events050.exact12971RawTerms
def group : MergeGroup := .relation 13064
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13064) (rhsResult := 12971)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25855⟩⟩) ⟨23466⟩ 12971) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23466⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨23466⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13065

namespace LeftMerge13066
def owner : Owner := ⟨.program ⟨214⟩, ⟨25856⟩⟩
def mergeEvent : Nat := 13066
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩] } }
def leftRaw : List Term := Proof.Events051.exact13057RawTerms
def rightRaw : List Term := Proof.Events050.exact12974RawTerms
def group : MergeGroup := .operator 13057 12974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13057) (leftOrdinal := 0)
    (rightResult := 12974) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25855⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13066

namespace LeftMerge13080
def owner : Owner := ⟨.program ⟨214⟩, ⟨19331⟩⟩
def mergeEvent : Nat := 13080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19328⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events051.exact13074RawTerms
def group : MergeGroup := .operator 6561 13074
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 13074) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19328⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19328⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13080

namespace LeftMerge13159
def owner : Owner := ⟨.program ⟨214⟩, ⟨13593⟩⟩
def mergeEvent : Nat := 13159
def frameStart : Nat := 13129
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events051.exact13155RawTerms
def rightRaw : List Term := Proof.Events051.exact13152RawTerms
def group : MergeGroup := .operator 13155 13152
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13155) (leftOrdinal := 0)
    (rightResult := 13152) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11233⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13159

namespace LeftMerge13189
def owner : Owner := ⟨.program ⟨214⟩, ⟨13681⟩⟩
def mergeEvent : Nat := 13189
def frameStart : Nat := 13129
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events051.exact13185RawTerms
def rightRaw : List Term := Proof.Events051.exact13183RawTerms
def group : MergeGroup := .operator 13185 13183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13185) (leftOrdinal := 0)
    (rightResult := 13183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13189

namespace LeftMerge13212
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def mergeEvent : Nat := 13212
def frameStart : Nat := 13129
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }
def leftRaw : List Term := Proof.Events051.exact13208RawTerms
def rightRaw : List Term := Proof.Events051.exact13205RawTerms
def group : MergeGroup := .operator 13208 13205
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13208) (leftOrdinal := 0)
    (rightResult := 13205) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7843⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13212

namespace LeftMerge13221
def owner : Owner := ⟨.program ⟨214⟩, ⟨25858⟩⟩
def mergeEvent : Nat := 13221
def frameStart : Nat := 13129
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩] } }
def leftRaw : List Term := Proof.Events051.exact13217RawTerms
def rightRaw : List Term := Proof.Events051.exact13174RawTerms
def group : MergeGroup := .operator 13217 13174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13217) (leftOrdinal := 1)
    (rightResult := 13174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25855⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13221

namespace LeftMerge13223
def owner : Owner := ⟨.program ⟨214⟩, ⟨25858⟩⟩
def mergeEvent : Nat := 13223
def frameStart : Nat := 13129
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23466⟩⟩] } }
def rhsRaw : List Term := Proof.Events051.exact13171RawTerms
def group : MergeGroup := .relation 13222
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13222) (rhsResult := 13171)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25855⟩⟩) ⟨23466⟩ 13171) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23466⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨23466⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13223

namespace LeftMerge13224
def owner : Owner := ⟨.program ⟨214⟩, ⟨25858⟩⟩
def mergeEvent : Nat := 13224
def frameStart : Nat := 13129
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩] } }
def leftRaw : List Term := Proof.Events051.exact13217RawTerms
def rightRaw : List Term := Proof.Events051.exact13174RawTerms
def group : MergeGroup := .operator 13217 13174
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13217) (leftOrdinal := 0)
    (rightResult := 13174) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25855⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13224

namespace LeftMerge13232
def owner : Owner := ⟨.program ⟨214⟩, ⟨15601⟩⟩
def mergeEvent : Nat := 13232
def frameStart : Nat := 13129
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15599⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events051.exact13185RawTerms
def rightRaw : List Term := Proof.Events051.exact13228RawTerms
def group : MergeGroup := .operator 13185 13228
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 13185) (leftOrdinal := 0)
    (rightResult := 13228) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15599⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15599⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13232

namespace LeftMerge13249
def owner : Owner := ⟨.program ⟨214⟩, ⟨19331⟩⟩
def mergeEvent : Nat := 13249
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23466⟩⟩] } }
def rhsRaw : List Term := Proof.Events051.exact13246RawTerms
def group : MergeGroup := .relation 13248
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13248) (rhsResult := 13246)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19328⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 13247 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19328⟩⟩]⟩) (none) 13246) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23466⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨11233⟩⟩, ⟨.program ⟨214⟩, ⟨13592⟩⟩], [⟨.program ⟨214⟩, ⟨23466⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge13249

namespace LeftMerge13250
def owner : Owner := ⟨.program ⟨214⟩, ⟨19331⟩⟩
def mergeEvent : Nat := 13250
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩] } }
def rhsRaw : List Term := Proof.Events051.exact13246RawTerms
def group : MergeGroup := .relation 13248
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 13248) (rhsResult := 13246)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19328⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 13247 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19328⟩⟩]⟩) (none) 13246) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6793⟩⟩, ⟨.program ⟨214⟩, ⟨7843⟩⟩, ⟨.program ⟨214⟩, ⟨25855⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge13250

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
