import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge17319
def owner : Owner := ⟨.program ⟨257⟩, ⟨49566⟩⟩
def mergeEvent : Nat := 17319
def frameStart : Nat := 17224
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17312RawTerms
def rightRaw : List Term := Proof.Events067.exact17269RawTerms
def group : MergeGroup := .operator 17312 17269
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17312) (leftOrdinal := 0)
    (rightResult := 17269) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49563⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17319

namespace LeftMerge17327
def owner : Owner := ⟨.program ⟨257⟩, ⟨48080⟩⟩
def mergeEvent : Nat := 17327
def frameStart : Nat := 17224
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17280RawTerms
def rightRaw : List Term := Proof.Events067.exact17323RawTerms
def group : MergeGroup := .operator 17280 17323
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17280) (leftOrdinal := 0)
    (rightResult := 17323) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17327

namespace LeftMerge17344
def owner : Owner := ⟨.program ⟨257⟩, ⟨48505⟩⟩
def mergeEvent : Nat := 17344
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49097⟩⟩] } }
def rhsRaw : List Term := Proof.Events067.exact17341RawTerms
def group : MergeGroup := .relation 17343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17343) (rhsResult := 17341)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 17342 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩) (none) 17341) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49097⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17344

namespace LeftMerge17345
def owner : Owner := ⟨.program ⟨257⟩, ⟨48505⟩⟩
def mergeEvent : Nat := 17345
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩] } }
def rhsRaw : List Term := Proof.Events067.exact17341RawTerms
def group : MergeGroup := .relation 17343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17343) (rhsResult := 17341)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 17342 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩) (none) 17341) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17345

namespace LeftMerge17346
def owner : Owner := ⟨.program ⟨257⟩, ⟨48505⟩⟩
def mergeEvent : Nat := 17346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events067.exact17341RawTerms
def group : MergeGroup := .relation 17343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17343) (rhsResult := 17341)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 17342 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩) (none) 17341) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17346

namespace LeftMerge17347
def owner : Owner := ⟨.program ⟨257⟩, ⟨48505⟩⟩
def mergeEvent : Nat := 17347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events067.exact17341RawTerms
def group : MergeGroup := .relation 17343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17343) (rhsResult := 17341)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 17342 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48502⟩⟩]⟩) (none) 17341) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17347

namespace LeftMerge17352
def owner : Owner := ⟨.program ⟨257⟩, ⟨49565⟩⟩
def mergeEvent : Nat := 17352
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49097⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17348RawTerms
def rightRaw : List Term := Proof.Events066.exact17147RawTerms
def group : MergeGroup := .operator 17348 17147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17348) (leftOrdinal := 2)
    (rightResult := 17147) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49097⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49097⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨14951⟩⟩, ⟨.program ⟨257⟩, ⟨47626⟩⟩], [⟨.program ⟨257⟩, ⟨49097⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17352

namespace LeftMerge17353
def owner : Owner := ⟨.program ⟨257⟩, ⟨49565⟩⟩
def mergeEvent : Nat := 17353
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17348RawTerms
def rightRaw : List Term := Proof.Events066.exact17147RawTerms
def group : MergeGroup := .operator 17348 17147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17348) (leftOrdinal := 1)
    (rightResult := 17147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49563⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17353

namespace LeftMerge17361
def owner : Owner := ⟨.program ⟨257⟩, ⟨49813⟩⟩
def mergeEvent : Nat := 17361
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17355RawTerms
def rightRaw : List Term := Proof.Events066.exact17037RawTerms
def group : MergeGroup := .operator 17355 17037
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17355) (leftOrdinal := 1)
    (rightResult := 17037) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49811⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17361

namespace LeftMerge17363
def owner : Owner := ⟨.program ⟨257⟩, ⟨49813⟩⟩
def mergeEvent : Nat := 17363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49223⟩⟩] } }
def rhsRaw : List Term := Proof.Events066.exact17034RawTerms
def group : MergeGroup := .relation 17362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17362) (rhsResult := 17034)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49811⟩⟩) ⟨49223⟩ 17034) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17363

namespace LeftMerge17364
def owner : Owner := ⟨.program ⟨257⟩, ⟨49813⟩⟩
def mergeEvent : Nat := 17364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17355RawTerms
def rightRaw : List Term := Proof.Events066.exact17037RawTerms
def group : MergeGroup := .operator 17355 17037
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17355) (leftOrdinal := 0)
    (rightResult := 17037) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49811⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17364

namespace LeftMerge17378
def owner : Owner := ⟨.program ⟨257⟩, ⟨48725⟩⟩
def mergeEvent : Nat := 17378
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48722⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events067.exact17372RawTerms
def group : MergeGroup := .operator 17169 17372
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 17372) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48722⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48722⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17378

namespace LeftMerge17499
def owner : Owner := ⟨.program ⟨257⟩, ⟨49472⟩⟩
def mergeEvent : Nat := 17499
def frameStart : Nat := 17433
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events068.exact17495RawTerms
def rightRaw : List Term := Proof.Events068.exact17493RawTerms
def group : MergeGroup := .operator 17495 17493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17495) (leftOrdinal := 0)
    (rightResult := 17493) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17499

namespace LeftMerge17511
def owner : Owner := ⟨.program ⟨257⟩, ⟨49812⟩⟩
def mergeEvent : Nat := 17511
def frameStart : Nat := 17433
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩] } }
def leftRaw : List Term := Proof.Events068.exact17507RawTerms
def rightRaw : List Term := Proof.Events068.exact17484RawTerms
def group : MergeGroup := .operator 17507 17484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17507) (leftOrdinal := 1)
    (rightResult := 17484) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49811⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17511

namespace LeftMerge17513
def owner : Owner := ⟨.program ⟨257⟩, ⟨49812⟩⟩
def mergeEvent : Nat := 17513
def frameStart : Nat := 17433
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48078⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49223⟩⟩] } }
def rhsRaw : List Term := Proof.Events068.exact17481RawTerms
def group : MergeGroup := .relation 17512
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 17512) (rhsResult := 17481)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49811⟩⟩) ⟨49223⟩ 17481) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48078⟩⟩], [⟨.program ⟨257⟩, ⟨49223⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge17513

namespace LeftMerge17514
def owner : Owner := ⟨.program ⟨257⟩, ⟨49812⟩⟩
def mergeEvent : Nat := 17514
def frameStart : Nat := 17433
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩] } }
def leftRaw : List Term := Proof.Events068.exact17507RawTerms
def rightRaw : List Term := Proof.Events068.exact17484RawTerms
def group : MergeGroup := .operator 17507 17484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17507) (leftOrdinal := 0)
    (rightResult := 17484) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49811⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨49811⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge17514

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
