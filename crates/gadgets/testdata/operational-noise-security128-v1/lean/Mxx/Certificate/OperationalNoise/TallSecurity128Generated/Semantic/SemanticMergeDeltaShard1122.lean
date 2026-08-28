import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge183328
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 183328
def frameStart : Nat := 183245
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events716.exact183324RawTerms
def rightRaw : List Term := Proof.Events716.exact183321RawTerms
def group : MergeGroup := .operator 183324 183321
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183324) (leftOrdinal := 0)
    (rightResult := 183321) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183328

namespace LeftMerge183337
def owner : Owner := ⟨.program ⟨257⟩, ⟨61495⟩⟩
def mergeEvent : Nat := 183337
def frameStart : Nat := 183245
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩] } }
def leftRaw : List Term := Proof.Events716.exact183333RawTerms
def rightRaw : List Term := Proof.Events715.exact183290RawTerms
def group : MergeGroup := .operator 183333 183290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183333) (leftOrdinal := 0)
    (rightResult := 183290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61492⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183337

namespace LeftMerge183338
def owner : Owner := ⟨.program ⟨257⟩, ⟨61495⟩⟩
def mergeEvent : Nat := 183338
def frameStart : Nat := 183245
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩] } }
def leftRaw : List Term := Proof.Events716.exact183333RawTerms
def rightRaw : List Term := Proof.Events715.exact183290RawTerms
def group : MergeGroup := .operator 183333 183290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183333) (leftOrdinal := 1)
    (rightResult := 183290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61492⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183338

namespace LeftMerge183340
def owner : Owner := ⟨.program ⟨257⟩, ⟨61495⟩⟩
def mergeEvent : Nat := 183340
def frameStart : Nat := 183245
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60967⟩⟩] } }
def rhsRaw : List Term := Proof.Events715.exact183287RawTerms
def group : MergeGroup := .relation 183339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183339) (rhsResult := 183287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61492⟩⟩) ⟨60967⟩ 183287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60967⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], [⟨.program ⟨257⟩, ⟨60967⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183340

namespace LeftMerge183348
def owner : Owner := ⟨.program ⟨257⟩, ⟨59854⟩⟩
def mergeEvent : Nat := 183348
def frameStart : Nat := 183245
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events716.exact183301RawTerms
def rightRaw : List Term := Proof.Events716.exact183344RawTerms
def group : MergeGroup := .operator 183301 183344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183301) (leftOrdinal := 0)
    (rightResult := 183344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59852⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183348

namespace LeftMerge183365
def owner : Owner := ⟨.program ⟨257⟩, ⟨60422⟩⟩
def mergeEvent : Nat := 183365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events716.exact183362RawTerms
def group : MergeGroup := .relation 183364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183364) (rhsResult := 183362)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 183363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60419⟩⟩]⟩) (none) 183362) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183365

namespace LeftMerge183366
def owner : Owner := ⟨.program ⟨257⟩, ⟨60422⟩⟩
def mergeEvent : Nat := 183366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩] } }
def rhsRaw : List Term := Proof.Events716.exact183362RawTerms
def group : MergeGroup := .relation 183364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183364) (rhsResult := 183362)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 183363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60419⟩⟩]⟩) (none) 183362) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183366

namespace LeftMerge183367
def owner : Owner := ⟨.program ⟨257⟩, ⟨60422⟩⟩
def mergeEvent : Nat := 183367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60967⟩⟩] } }
def rhsRaw : List Term := Proof.Events716.exact183362RawTerms
def group : MergeGroup := .relation 183364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183364) (rhsResult := 183362)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 183363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60419⟩⟩]⟩) (none) 183362) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60967⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], [⟨.program ⟨257⟩, ⟨60967⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183367

namespace LeftMerge183368
def owner : Owner := ⟨.program ⟨257⟩, ⟨60422⟩⟩
def mergeEvent : Nat := 183368
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨59852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events716.exact183362RawTerms
def group : MergeGroup := .relation 183364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183364) (rhsResult := 183362)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 183363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60419⟩⟩]⟩) (none) 183362) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨59852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183368

namespace LeftMerge183373
def owner : Owner := ⟨.program ⟨257⟩, ⟨61494⟩⟩
def mergeEvent : Nat := 183373
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60967⟩⟩] } }
def leftRaw : List Term := Proof.Events716.exact183369RawTerms
def rightRaw : List Term := Proof.Events715.exact183183RawTerms
def group : MergeGroup := .operator 183369 183183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183369) (leftOrdinal := 2)
    (rightResult := 183183) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60967⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60967⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨25286⟩⟩, ⟨.program ⟨257⟩, ⟨59566⟩⟩], [⟨.program ⟨257⟩, ⟨60967⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183373

namespace LeftMerge183374
def owner : Owner := ⟨.program ⟨257⟩, ⟨61494⟩⟩
def mergeEvent : Nat := 183374
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩] } }
def leftRaw : List Term := Proof.Events716.exact183369RawTerms
def rightRaw : List Term := Proof.Events715.exact183183RawTerms
def group : MergeGroup := .operator 183369 183183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183369) (leftOrdinal := 1)
    (rightResult := 183183) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61492⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183374

namespace LeftMerge183382
def owner : Owner := ⟨.program ⟨257⟩, ⟨61987⟩⟩
def mergeEvent : Nat := 183382
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61985⟩⟩] } }
def leftRaw : List Term := Proof.Events716.exact183376RawTerms
def rightRaw : List Term := Proof.Events715.exact183099RawTerms
def group : MergeGroup := .operator 183376 183099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183376) (leftOrdinal := 0)
    (rightResult := 183099) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61985⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183382

namespace LeftMerge183383
def owner : Owner := ⟨.program ⟨257⟩, ⟨61987⟩⟩
def mergeEvent : Nat := 183383
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨59852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61985⟩⟩] } }
def leftRaw : List Term := Proof.Events716.exact183376RawTerms
def rightRaw : List Term := Proof.Events715.exact183099RawTerms
def group : MergeGroup := .operator 183376 183099
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183376) (leftOrdinal := 1)
    (rightResult := 183099) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨59852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61985⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨59852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61985⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183383

namespace LeftMerge183385
def owner : Owner := ⟨.program ⟨257⟩, ⟨61987⟩⟩
def mergeEvent : Nat := 183385
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨59852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61128⟩⟩] } }
def rhsRaw : List Term := Proof.Events715.exact183096RawTerms
def group : MergeGroup := .relation 183384
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 183384) (rhsResult := 183096)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨59852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61985⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61985⟩⟩) ⟨61128⟩ 183096) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61128⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨59852⟩⟩], [⟨.program ⟨257⟩, ⟨61128⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge183385

namespace LeftMerge183399
def owner : Owner := ⟨.program ⟨257⟩, ⟨60759⟩⟩
def mergeEvent : Nat := 183399
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60756⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events716.exact183393RawTerms
def group : MergeGroup := .operator 178370 183393
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 183393) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60756⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183399

namespace LeftMerge183520
def owner : Owner := ⟨.program ⟨257⟩, ⟨61320⟩⟩
def mergeEvent : Nat := 183520
def frameStart : Nat := 183454
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59852⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events716.exact183516RawTerms
def rightRaw : List Term := Proof.Events716.exact183514RawTerms
def group : MergeGroup := .operator 183516 183514
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 183516) (leftOrdinal := 0)
    (rightResult := 183514) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59852⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59852⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge183520

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
