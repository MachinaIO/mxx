import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge62197
def owner : Owner := ⟨.program ⟨257⟩, ⟨47525⟩⟩
def mergeEvent : Nat := 62197
def frameStart : Nat := 62116
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46684⟩⟩] } }
def rhsRaw : List Term := Proof.Events242.exact62164RawTerms
def group : MergeGroup := .relation 62196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62196) (rhsResult := 62164)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47524⟩⟩) ⟨46684⟩ 62164) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46684⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨46684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62197

namespace LeftMerge62205
def owner : Owner := ⟨.program ⟨257⟩, ⟨45775⟩⟩
def mergeEvent : Nat := 62205
def frameStart : Nat := 62116
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62178RawTerms
def rightRaw : List Term := Proof.Events242.exact62201RawTerms
def group : MergeGroup := .operator 62178 62201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62178) (leftOrdinal := 0)
    (rightResult := 62201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62205

namespace LeftMerge62222
def owner : Owner := ⟨.program ⟨257⟩, ⟨46359⟩⟩
def mergeEvent : Nat := 62222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62219RawTerms
def group : MergeGroup := .relation 62221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62221) (rhsResult := 62219)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩]⟩) (none) 62219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62222

namespace LeftMerge62223
def owner : Owner := ⟨.program ⟨257⟩, ⟨46359⟩⟩
def mergeEvent : Nat := 62223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62219RawTerms
def group : MergeGroup := .relation 62221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62221) (rhsResult := 62219)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩]⟩) (none) 62219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62223

namespace LeftMerge62224
def owner : Owner := ⟨.program ⟨257⟩, ⟨46359⟩⟩
def mergeEvent : Nat := 62224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46684⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62219RawTerms
def group : MergeGroup := .relation 62221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62221) (rhsResult := 62219)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩]⟩) (none) 62219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46684⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨46684⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62224

namespace LeftMerge62225
def owner : Owner := ⟨.program ⟨257⟩, ⟨46359⟩⟩
def mergeEvent : Nat := 62225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events243.exact62219RawTerms
def group : MergeGroup := .relation 62221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62221) (rhsResult := 62219)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩]⟩) (none) 62219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62225

namespace LeftMerge62230
def owner : Owner := ⟨.program ⟨257⟩, ⟨47527⟩⟩
def mergeEvent : Nat := 62230
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62226RawTerms
def rightRaw : List Term := Proof.Events242.exact62048RawTerms
def group : MergeGroup := .operator 62226 62048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62226) (leftOrdinal := 0)
    (rightResult := 62048) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62230

namespace LeftMerge62231
def owner : Owner := ⟨.program ⟨257⟩, ⟨47527⟩⟩
def mergeEvent : Nat := 62231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46684⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62226RawTerms
def rightRaw : List Term := Proof.Events242.exact62048RawTerms
def group : MergeGroup := .operator 62226 62048
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62226) (leftOrdinal := 2)
    (rightResult := 62048) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46684⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46684⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨46684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62231

namespace LeftMerge62257
def owner : Owner := ⟨.program ⟨257⟩, ⟨42645⟩⟩
def mergeEvent : Nat := 62257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2384RawTerms
def rightRaw : List Term := Proof.Events239.exact61278RawTerms
def group : MergeGroup := .operator 2384 61278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2384) (leftOrdinal := 0)
    (rightResult := 61278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62257

namespace LeftMerge62262
def owner : Owner := ⟨.program ⟨257⟩, ⟨10765⟩⟩
def mergeEvent : Nat := 62262
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61148RawTerms
def rightRaw : List Term := Proof.Events070.exact18082RawTerms
def group : MergeGroup := .operator 61148 18082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61148) (leftOrdinal := 0)
    (rightResult := 18082) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62262

namespace LeftMerge62279
def owner : Owner := ⟨.program ⟨257⟩, ⟨42648⟩⟩
def mergeEvent : Nat := 62279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62273RawTerms
def rightRaw : List Term := Proof.Events009.exact2387RawTerms
def group : MergeGroup := .operator 62273 2387
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62273) (leftOrdinal := 1)
    (rightResult := 2387) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩, ⟨.program ⟨257⟩, ⟨42642⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62279

namespace LeftMerge62280
def owner : Owner := ⟨.program ⟨257⟩, ⟨42648⟩⟩
def mergeEvent : Nat := 62280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62273RawTerms
def rightRaw : List Term := Proof.Events009.exact2387RawTerms
def group : MergeGroup := .operator 62273 2387
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62273) (leftOrdinal := 0)
    (rightResult := 2387) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62280

namespace LeftMerge62285
def owner : Owner := ⟨.program ⟨257⟩, ⟨14587⟩⟩
def mergeEvent : Nat := 62285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events009.exact2387RawTerms
def rightRaw : List Term := Proof.Events239.exact61278RawTerms
def group : MergeGroup := .operator 2387 61278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 2387) (leftOrdinal := 0)
    (rightResult := 61278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62285

namespace LeftMerge62290
def owner : Owner := ⟨.program ⟨257⟩, ⟨10782⟩⟩
def mergeEvent : Nat := 62290
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61148RawTerms
def rightRaw : List Term := Proof.Events070.exact18123RawTerms
def group : MergeGroup := .operator 61148 18123
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61148) (leftOrdinal := 0)
    (rightResult := 18123) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7300⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7300⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62290

namespace LeftMerge62307
def owner : Owner := ⟨.program ⟨257⟩, ⟨14590⟩⟩
def mergeEvent : Nat := 62307
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩] } }
def leftRaw : List Term := Proof.Events243.exact62301RawTerms
def rightRaw : List Term := Proof.Events070.exact18112RawTerms
def group : MergeGroup := .operator 62301 18112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62301) (leftOrdinal := 1)
    (rightResult := 18112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9559⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62307

namespace LeftMerge62309
def owner : Owner := ⟨.program ⟨257⟩, ⟨14590⟩⟩
def mergeEvent : Nat := 62309
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }
def rhsRaw : List Term := Proof.Events070.exact18082RawTerms
def group : MergeGroup := .relation 62308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62308) (rhsResult := 18082)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9559⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9559⟩⟩) ⟨7283⟩ 18082) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7283⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14586⟩⟩], [⟨.program ⟨257⟩, ⟨7283⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62309

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
