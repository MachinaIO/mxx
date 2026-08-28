import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge182177
def owner : Owner := ⟨.program ⟨257⟩, ⟨65530⟩⟩
def mergeEvent : Nat := 182177
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events033.exact8509RawTerms
def rightRaw : List Term := Proof.Events696.exact178278RawTerms
def group : MergeGroup := .operator 8509 178278
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 8509) (leftOrdinal := 0)
    (rightResult := 178278) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182177

namespace LeftMerge182182
def owner : Owner := ⟨.program ⟨257⟩, ⟨8942⟩⟩
def mergeEvent : Nat := 182182
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }
def leftRaw : List Term := Proof.Events695.exact178148RawTerms
def rightRaw : List Term := Proof.Events082.exact21129RawTerms
def group : MergeGroup := .operator 178148 21129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178148) (leftOrdinal := 0)
    (rightResult := 21129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182182

namespace LeftMerge182199
def owner : Owner := ⟨.program ⟨257⟩, ⟨65533⟩⟩
def mergeEvent : Nat := 182199
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events711.exact182193RawTerms
def rightRaw : List Term := Proof.Events082.exact21118RawTerms
def group : MergeGroup := .operator 182193 21118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182193) (leftOrdinal := 1)
    (rightResult := 21118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge182199

namespace LeftMerge182201
def owner : Owner := ⟨.program ⟨257⟩, ⟨65533⟩⟩
def mergeEvent : Nat := 182201
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def rhsRaw : List Term := Proof.Events082.exact21088RawTerms
def group : MergeGroup := .relation 182200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 182200) (rhsResult := 21088)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9541⟩⟩) ⟨7276⟩ 21088) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge182201

namespace LeftMerge182202
def owner : Owner := ⟨.program ⟨257⟩, ⟨65533⟩⟩
def mergeEvent : Nat := 182202
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events711.exact182193RawTerms
def rightRaw : List Term := Proof.Events082.exact21118RawTerms
def group : MergeGroup := .operator 182193 21118
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182193) (leftOrdinal := 0)
    (rightResult := 21118) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182202

namespace LeftMerge182207
def owner : Owner := ⟨.program ⟨257⟩, ⟨65534⟩⟩
def mergeEvent : Nat := 182207
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }
def leftRaw : List Term := Proof.Events711.exact182203RawTerms
def rightRaw : List Term := Proof.Events711.exact182173RawTerms
def group : MergeGroup := .operator 182203 182173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182203) (leftOrdinal := 1)
    (rightResult := 182173) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7276⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨7276⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182207

namespace LeftMerge182215
def owner : Owner := ⟨.program ⟨257⟩, ⟨69274⟩⟩
def mergeEvent : Nat := 182215
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩] } }
def leftRaw : List Term := Proof.Events711.exact182209RawTerms
def rightRaw : List Term := Proof.Events711.exact182145RawTerms
def group : MergeGroup := .operator 182209 182145
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182209) (leftOrdinal := 1)
    (rightResult := 182145) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge182215

namespace LeftMerge182217
def owner : Owner := ⟨.program ⟨257⟩, ⟨69274⟩⟩
def mergeEvent : Nat := 182217
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68548⟩⟩] } }
def rhsRaw : List Term := Proof.Events711.exact182142RawTerms
def group : MergeGroup := .relation 182216
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 182216) (rhsResult := 182142)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69273⟩⟩) ⟨68548⟩ 182142) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68548⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨68548⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge182217

namespace LeftMerge182218
def owner : Owner := ⟨.program ⟨257⟩, ⟨69274⟩⟩
def mergeEvent : Nat := 182218
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩] } }
def leftRaw : List Term := Proof.Events711.exact182209RawTerms
def rightRaw : List Term := Proof.Events711.exact182145RawTerms
def group : MergeGroup := .operator 182209 182145
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182209) (leftOrdinal := 0)
    (rightResult := 182145) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182218

namespace LeftMerge182232
def owner : Owner := ⟨.program ⟨257⟩, ⟨67803⟩⟩
def mergeEvent : Nat := 182232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67800⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events711.exact182226RawTerms
def group : MergeGroup := .operator 178370 182226
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 182226) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67800⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67800⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182232

namespace LeftMerge182311
def owner : Owner := ⟨.program ⟨257⟩, ⟨65527⟩⟩
def mergeEvent : Nat := 182311
def frameStart : Nat := 182281
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events712.exact182307RawTerms
def rightRaw : List Term := Proof.Events712.exact182304RawTerms
def group : MergeGroup := .operator 182307 182304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182307) (leftOrdinal := 0)
    (rightResult := 182304) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25766⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182311

namespace LeftMerge182341
def owner : Owner := ⟨.program ⟨257⟩, ⟨68941⟩⟩
def mergeEvent : Nat := 182341
def frameStart : Nat := 182281
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events712.exact182337RawTerms
def rightRaw : List Term := Proof.Events712.exact182335RawTerms
def group : MergeGroup := .operator 182337 182335
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182337) (leftOrdinal := 0)
    (rightResult := 182335) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182341

namespace LeftMerge182364
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def mergeEvent : Nat := 182364
def frameStart : Nat := 182281
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events712.exact182360RawTerms
def rightRaw : List Term := Proof.Events712.exact182357RawTerms
def group : MergeGroup := .operator 182360 182357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182360) (leftOrdinal := 0)
    (rightResult := 182357) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182364

namespace LeftMerge182373
def owner : Owner := ⟨.program ⟨257⟩, ⟨69276⟩⟩
def mergeEvent : Nat := 182373
def frameStart : Nat := 182281
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩] } }
def leftRaw : List Term := Proof.Events712.exact182369RawTerms
def rightRaw : List Term := Proof.Events712.exact182326RawTerms
def group : MergeGroup := .operator 182369 182326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182369) (leftOrdinal := 0)
    (rightResult := 182326) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69273⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge182373

namespace LeftMerge182374
def owner : Owner := ⟨.program ⟨257⟩, ⟨69276⟩⟩
def mergeEvent : Nat := 182374
def frameStart : Nat := 182281
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩] } }
def leftRaw : List Term := Proof.Events712.exact182369RawTerms
def rightRaw : List Term := Proof.Events712.exact182326RawTerms
def group : MergeGroup := .operator 182369 182326
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 182369) (leftOrdinal := 1)
    (rightResult := 182326) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69273⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge182374

namespace LeftMerge182376
def owner : Owner := ⟨.program ⟨257⟩, ⟨69276⟩⟩
def mergeEvent : Nat := 182376
def frameStart : Nat := 182281
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68548⟩⟩] } }
def rhsRaw : List Term := Proof.Events712.exact182323RawTerms
def group : MergeGroup := .relation 182375
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 182375) (rhsResult := 182323)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69273⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69273⟩⟩) ⟨68548⟩ 182323) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68548⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25766⟩⟩, ⟨.program ⟨257⟩, ⟨65526⟩⟩], [⟨.program ⟨257⟩, ⟨68548⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge182376

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
