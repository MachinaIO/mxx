import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge40323
def owner : Owner := ⟨.program ⟨214⟩, ⟨16228⟩⟩
def mergeEvent : Nat := 40323
def frameStart : Nat := 40257
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40319RawTerms
def rightRaw : List Term := Proof.Events157.exact40317RawTerms
def group : MergeGroup := .operator 40319 40317
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40319) (leftOrdinal := 0)
    (rightResult := 40317) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40323

namespace LeftMerge40335
def owner : Owner := ⟨.program ⟨214⟩, ⟨28327⟩⟩
def mergeEvent : Nat := 40335
def frameStart : Nat := 40257
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40331RawTerms
def rightRaw : List Term := Proof.Events157.exact40308RawTerms
def group : MergeGroup := .operator 40331 40308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40331) (leftOrdinal := 0)
    (rightResult := 40308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28326⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40335

namespace LeftMerge40336
def owner : Owner := ⟨.program ⟨214⟩, ⟨28327⟩⟩
def mergeEvent : Nat := 40336
def frameStart : Nat := 40257
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40331RawTerms
def rightRaw : List Term := Proof.Events157.exact40308RawTerms
def group : MergeGroup := .operator 40331 40308
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40331) (leftOrdinal := 1)
    (rightResult := 40308) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28326⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40336

namespace LeftMerge40338
def owner : Owner := ⟨.program ⟨214⟩, ⟨28327⟩⟩
def mergeEvent : Nat := 40338
def frameStart : Nat := 40257
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24294⟩⟩] } }
def rhsRaw : List Term := Proof.Events157.exact40305RawTerms
def group : MergeGroup := .relation 40337
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40337) (rhsResult := 40305)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28326⟩⟩) ⟨24294⟩ 40305) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24294⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24294⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40338

namespace LeftMerge40346
def owner : Owner := ⟨.program ⟨214⟩, ⟨18377⟩⟩
def mergeEvent : Nat := 40346
def frameStart : Nat := 40257
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40319RawTerms
def rightRaw : List Term := Proof.Events157.exact40342RawTerms
def group : MergeGroup := .operator 40319 40342
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40319) (leftOrdinal := 0)
    (rightResult := 40342) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40346

namespace LeftMerge40363
def owner : Owner := ⟨.program ⟨214⟩, ⟨21699⟩⟩
def mergeEvent : Nat := 40363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }
def rhsRaw : List Term := Proof.Events157.exact40360RawTerms
def group : MergeGroup := .relation 40362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40362) (rhsResult := 40360)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40361 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩) (none) 40360) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40363

namespace LeftMerge40364
def owner : Owner := ⟨.program ⟨214⟩, ⟨21699⟩⟩
def mergeEvent : Nat := 40364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩] } }
def rhsRaw : List Term := Proof.Events157.exact40360RawTerms
def group : MergeGroup := .relation 40362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40362) (rhsResult := 40360)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40361 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩) (none) 40360) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40364

namespace LeftMerge40365
def owner : Owner := ⟨.program ⟨214⟩, ⟨21699⟩⟩
def mergeEvent : Nat := 40365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24294⟩⟩] } }
def rhsRaw : List Term := Proof.Events157.exact40360RawTerms
def group : MergeGroup := .relation 40362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40362) (rhsResult := 40360)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40361 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩) (none) 40360) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24294⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24294⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40365

namespace LeftMerge40366
def owner : Owner := ⟨.program ⟨214⟩, ⟨21699⟩⟩
def mergeEvent : Nat := 40366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events157.exact40360RawTerms
def group : MergeGroup := .relation 40362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 40362) (rhsResult := 40360)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 40361 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21696⟩⟩]⟩) (none) 40360) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40366

namespace LeftMerge40371
def owner : Owner := ⟨.program ⟨214⟩, ⟨28329⟩⟩
def mergeEvent : Nat := 40371
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40367RawTerms
def rightRaw : List Term := Proof.Events156.exact40189RawTerms
def group : MergeGroup := .operator 40367 40189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40367) (leftOrdinal := 0)
    (rightResult := 40189) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28326⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40371

namespace LeftMerge40372
def owner : Owner := ⟨.program ⟨214⟩, ⟨28329⟩⟩
def mergeEvent : Nat := 40372
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24294⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40367RawTerms
def rightRaw : List Term := Proof.Events156.exact40189RawTerms
def group : MergeGroup := .operator 40367 40189
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40367) (leftOrdinal := 2)
    (rightResult := 40189) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24294⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24294⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16186⟩⟩], [⟨.program ⟨214⟩, ⟨24294⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40372

namespace LeftMerge40398
def owner : Owner := ⟨.program ⟨214⟩, ⟨11562⟩⟩
def mergeEvent : Nat := 40398
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events007.exact1797RawTerms
def rightRaw : List Term := Proof.Events140.exact36045RawTerms
def group : MergeGroup := .operator 1797 36045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1797) (leftOrdinal := 0)
    (rightResult := 36045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11561⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11561⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40398

namespace LeftMerge40403
def owner : Owner := ⟨.program ⟨214⟩, ⟨7312⟩⟩
def mergeEvent : Nat := 40403
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def leftRaw : List Term := Proof.Events140.exact35915RawTerms
def rightRaw : List Term := Proof.Events042.exact10981RawTerms
def group : MergeGroup := .operator 35915 10981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35915) (leftOrdinal := 0)
    (rightResult := 10981) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40403

namespace LeftMerge40420
def owner : Owner := ⟨.program ⟨214⟩, ⟨14445⟩⟩
def mergeEvent : Nat := 40420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11561⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40414RawTerms
def rightRaw : List Term := Proof.Events007.exact1800RawTerms
def group : MergeGroup := .operator 40414 1800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40414) (leftOrdinal := 1)
    (rightResult := 1800) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11561⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14442⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨11561⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge40420

namespace LeftMerge40421
def owner : Owner := ⟨.program ⟨214⟩, ⟨14445⟩⟩
def mergeEvent : Nat := 40421
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def leftRaw : List Term := Proof.Events157.exact40414RawTerms
def rightRaw : List Term := Proof.Events007.exact1800RawTerms
def group : MergeGroup := .operator 40414 1800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 40414) (leftOrdinal := 0)
    (rightResult := 1800) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14442⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40421

namespace LeftMerge40426
def owner : Owner := ⟨.program ⟨214⟩, ⟨14446⟩⟩
def mergeEvent : Nat := 40426
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events007.exact1800RawTerms
def rightRaw : List Term := Proof.Events140.exact36045RawTerms
def group : MergeGroup := .operator 1800 36045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1800) (leftOrdinal := 0)
    (rightResult := 36045) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14442⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨14442⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge40426

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
