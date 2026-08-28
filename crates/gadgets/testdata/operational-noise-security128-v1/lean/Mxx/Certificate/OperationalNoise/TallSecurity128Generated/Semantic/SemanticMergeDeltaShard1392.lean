import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge226107
def owner : Owner := ⟨.program ⟨257⟩, ⟨67763⟩⟩
def mergeEvent : Nat := 226107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩] } }
def leftRaw : List Term := Proof.Events868.exact222245RawTerms
def rightRaw : List Term := Proof.Events883.exact226101RawTerms
def group : MergeGroup := .operator 222245 226101
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 222245) (leftOrdinal := 0)
    (rightResult := 226101) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67760⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226107

namespace LeftMerge226186
def owner : Owner := ⟨.program ⟨257⟩, ⟨65419⟩⟩
def mergeEvent : Nat := 226186
def frameStart : Nat := 226156
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events883.exact226182RawTerms
def rightRaw : List Term := Proof.Events883.exact226179RawTerms
def group : MergeGroup := .operator 226182 226179
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226182) (leftOrdinal := 0)
    (rightResult := 226179) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25718⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226186

namespace LeftMerge226216
def owner : Owner := ⟨.program ⟨257⟩, ⟨68925⟩⟩
def mergeEvent : Nat := 226216
def frameStart : Nat := 226156
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events883.exact226212RawTerms
def rightRaw : List Term := Proof.Events883.exact226210RawTerms
def group : MergeGroup := .operator 226212 226210
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226212) (leftOrdinal := 0)
    (rightResult := 226210) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226216

namespace LeftMerge226239
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def mergeEvent : Nat := 226239
def frameStart : Nat := 226156
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }
def leftRaw : List Term := Proof.Events883.exact226235RawTerms
def rightRaw : List Term := Proof.Events883.exact226232RawTerms
def group : MergeGroup := .operator 226235 226232
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226235) (leftOrdinal := 0)
    (rightResult := 226232) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9541⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226239

namespace LeftMerge226248
def owner : Owner := ⟨.program ⟨257⟩, ⟨69232⟩⟩
def mergeEvent : Nat := 226248
def frameStart : Nat := 226156
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩] } }
def leftRaw : List Term := Proof.Events883.exact226244RawTerms
def rightRaw : List Term := Proof.Events883.exact226201RawTerms
def group : MergeGroup := .operator 226244 226201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226244) (leftOrdinal := 0)
    (rightResult := 226201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69229⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226248

namespace LeftMerge226249
def owner : Owner := ⟨.program ⟨257⟩, ⟨69232⟩⟩
def mergeEvent : Nat := 226249
def frameStart : Nat := 226156
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩] } }
def leftRaw : List Term := Proof.Events883.exact226244RawTerms
def rightRaw : List Term := Proof.Events883.exact226201RawTerms
def group : MergeGroup := .operator 226244 226201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226244) (leftOrdinal := 1)
    (rightResult := 226201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69229⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226249

namespace LeftMerge226251
def owner : Owner := ⟨.program ⟨257⟩, ⟨69232⟩⟩
def mergeEvent : Nat := 226251
def frameStart : Nat := 226156
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68524⟩⟩] } }
def rhsRaw : List Term := Proof.Events883.exact226198RawTerms
def group : MergeGroup := .relation 226250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226250) (rhsResult := 226198)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69229⟩⟩) ⟨68524⟩ 226198) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68524⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], [⟨.program ⟨257⟩, ⟨68524⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226251

namespace LeftMerge226259
def owner : Owner := ⟨.program ⟨257⟩, ⟨65782⟩⟩
def mergeEvent : Nat := 226259
def frameStart : Nat := 226156
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events883.exact226212RawTerms
def rightRaw : List Term := Proof.Events883.exact226255RawTerms
def group : MergeGroup := .operator 226212 226255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226212) (leftOrdinal := 0)
    (rightResult := 226255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65780⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226259

namespace LeftMerge226276
def owner : Owner := ⟨.program ⟨257⟩, ⟨67763⟩⟩
def mergeEvent : Nat := 226276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }
def rhsRaw : List Term := Proof.Events883.exact226273RawTerms
def group : MergeGroup := .relation 226275
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226275) (rhsResult := 226273)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 226274 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩]⟩) (none) 226273) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226276

namespace LeftMerge226277
def owner : Owner := ⟨.program ⟨257⟩, ⟨67763⟩⟩
def mergeEvent : Nat := 226277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩] } }
def rhsRaw : List Term := Proof.Events883.exact226273RawTerms
def group : MergeGroup := .relation 226275
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226275) (rhsResult := 226273)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 226274 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩]⟩) (none) 226273) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226277

namespace LeftMerge226278
def owner : Owner := ⟨.program ⟨257⟩, ⟨67763⟩⟩
def mergeEvent : Nat := 226278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68524⟩⟩] } }
def rhsRaw : List Term := Proof.Events883.exact226273RawTerms
def group : MergeGroup := .relation 226275
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226275) (rhsResult := 226273)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 226274 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩]⟩) (none) 226273) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68524⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], [⟨.program ⟨257⟩, ⟨68524⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226278

namespace LeftMerge226279
def owner : Owner := ⟨.program ⟨257⟩, ⟨67763⟩⟩
def mergeEvent : Nat := 226279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨65780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events883.exact226273RawTerms
def group : MergeGroup := .relation 226275
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 226275) (rhsResult := 226273)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 226274 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67760⟩⟩]⟩) (none) 226273) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226279

namespace LeftMerge226284
def owner : Owner := ⟨.program ⟨257⟩, ⟨69231⟩⟩
def mergeEvent : Nat := 226284
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68524⟩⟩] } }
def leftRaw : List Term := Proof.Events883.exact226280RawTerms
def rightRaw : List Term := Proof.Events883.exact226094RawTerms
def group : MergeGroup := .operator 226280 226094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226280) (leftOrdinal := 2)
    (rightResult := 226094) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68524⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68524⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨25718⟩⟩, ⟨.program ⟨257⟩, ⟨65418⟩⟩], [⟨.program ⟨257⟩, ⟨68524⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226284

namespace LeftMerge226285
def owner : Owner := ⟨.program ⟨257⟩, ⟨69231⟩⟩
def mergeEvent : Nat := 226285
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩] } }
def leftRaw : List Term := Proof.Events883.exact226280RawTerms
def rightRaw : List Term := Proof.Events883.exact226094RawTerms
def group : MergeGroup := .operator 226280 226094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226280) (leftOrdinal := 1)
    (rightResult := 226094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69229⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226285

namespace LeftMerge226293
def owner : Owner := ⟨.program ⟨257⟩, ⟨70100⟩⟩
def mergeEvent : Nat := 226293
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩] } }
def leftRaw : List Term := Proof.Events883.exact226287RawTerms
def rightRaw : List Term := Proof.Events882.exact226010RawTerms
def group : MergeGroup := .operator 226287 226010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226287) (leftOrdinal := 0)
    (rightResult := 226010) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70098⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge226293

namespace LeftMerge226294
def owner : Owner := ⟨.program ⟨257⟩, ⟨70100⟩⟩
def mergeEvent : Nat := 226294
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨65780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩] } }
def leftRaw : List Term := Proof.Events883.exact226287RawTerms
def rightRaw : List Term := Proof.Events882.exact226010RawTerms
def group : MergeGroup := .operator 226287 226010
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 226287) (leftOrdinal := 1)
    (rightResult := 226010) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨65780⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70098⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5243⟩⟩, ⟨.program ⟨257⟩, ⟨65780⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70098⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge226294

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
