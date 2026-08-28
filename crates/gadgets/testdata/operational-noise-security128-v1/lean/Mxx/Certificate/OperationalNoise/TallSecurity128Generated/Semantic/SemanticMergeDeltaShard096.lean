import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge19164
def owner : Owner := ⟨.program ⟨257⟩, ⟨38844⟩⟩
def mergeEvent : Nat := 19164
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38377⟩⟩] } }
def rhsRaw : List Term := Proof.Events074.exact19070RawTerms
def group : MergeGroup := .relation 19163
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19163) (rhsResult := 19070)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38843⟩⟩) ⟨38377⟩ 19070) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38377⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], [⟨.program ⟨257⟩, ⟨38377⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19164

namespace LeftMerge19165
def owner : Owner := ⟨.program ⟨257⟩, ⟨38844⟩⟩
def mergeEvent : Nat := 19165
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩] } }
def leftRaw : List Term := Proof.Events074.exact19156RawTerms
def rightRaw : List Term := Proof.Events074.exact19073RawTerms
def group : MergeGroup := .operator 19156 19073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19156) (leftOrdinal := 0)
    (rightResult := 19073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38843⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19165

namespace LeftMerge19179
def owner : Owner := ⟨.program ⟨257⟩, ⟨37785⟩⟩
def mergeEvent : Nat := 19179
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events074.exact19173RawTerms
def group : MergeGroup := .operator 17169 19173
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 19173) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37782⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19179

namespace LeftMerge19258
def owner : Owner := ⟨.program ⟨257⟩, ⟨36907⟩⟩
def mergeEvent : Nat := 19258
def frameStart : Nat := 19228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events075.exact19254RawTerms
def rightRaw : List Term := Proof.Events075.exact19251RawTerms
def group : MergeGroup := .operator 19254 19251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19254) (leftOrdinal := 0)
    (rightResult := 19251) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13751⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19258

namespace LeftMerge19288
def owner : Owner := ⟨.program ⟨257⟩, ⟨38672⟩⟩
def mergeEvent : Nat := 19288
def frameStart : Nat := 19228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events075.exact19284RawTerms
def rightRaw : List Term := Proof.Events075.exact19282RawTerms
def group : MergeGroup := .operator 19284 19282
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19284) (leftOrdinal := 0)
    (rightResult := 19282) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19288

namespace LeftMerge19311
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 19311
def frameStart : Nat := 19228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events075.exact19307RawTerms
def rightRaw : List Term := Proof.Events075.exact19304RawTerms
def group : MergeGroup := .operator 19307 19304
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19307) (leftOrdinal := 0)
    (rightResult := 19304) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19311

namespace LeftMerge19320
def owner : Owner := ⟨.program ⟨257⟩, ⟨38846⟩⟩
def mergeEvent : Nat := 19320
def frameStart : Nat := 19228
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩] } }
def leftRaw : List Term := Proof.Events075.exact19316RawTerms
def rightRaw : List Term := Proof.Events075.exact19273RawTerms
def group : MergeGroup := .operator 19316 19273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19316) (leftOrdinal := 1)
    (rightResult := 19273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38843⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19320

namespace LeftMerge19322
def owner : Owner := ⟨.program ⟨257⟩, ⟨38846⟩⟩
def mergeEvent : Nat := 19322
def frameStart : Nat := 19228
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38377⟩⟩] } }
def rhsRaw : List Term := Proof.Events075.exact19270RawTerms
def group : MergeGroup := .relation 19321
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19321) (rhsResult := 19270)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38843⟩⟩) ⟨38377⟩ 19270) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38377⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], [⟨.program ⟨257⟩, ⟨38377⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19322

namespace LeftMerge19323
def owner : Owner := ⟨.program ⟨257⟩, ⟨38846⟩⟩
def mergeEvent : Nat := 19323
def frameStart : Nat := 19228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩] } }
def leftRaw : List Term := Proof.Events075.exact19316RawTerms
def rightRaw : List Term := Proof.Events075.exact19273RawTerms
def group : MergeGroup := .operator 19316 19273
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19316) (leftOrdinal := 0)
    (rightResult := 19273) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38843⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19323

namespace LeftMerge19331
def owner : Owner := ⟨.program ⟨257⟩, ⟨37360⟩⟩
def mergeEvent : Nat := 19331
def frameStart : Nat := 19228
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events075.exact19284RawTerms
def rightRaw : List Term := Proof.Events075.exact19327RawTerms
def group : MergeGroup := .operator 19284 19327
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19284) (leftOrdinal := 0)
    (rightResult := 19327) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37358⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19331

namespace LeftMerge19348
def owner : Owner := ⟨.program ⟨257⟩, ⟨37785⟩⟩
def mergeEvent : Nat := 19348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38377⟩⟩] } }
def rhsRaw : List Term := Proof.Events075.exact19345RawTerms
def group : MergeGroup := .relation 19347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19347) (rhsResult := 19345)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩]⟩) (none) 19345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38377⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], [⟨.program ⟨257⟩, ⟨38377⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19348

namespace LeftMerge19349
def owner : Owner := ⟨.program ⟨257⟩, ⟨37785⟩⟩
def mergeEvent : Nat := 19349
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩] } }
def rhsRaw : List Term := Proof.Events075.exact19345RawTerms
def group : MergeGroup := .relation 19347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19347) (rhsResult := 19345)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩]⟩) (none) 19345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19349

namespace LeftMerge19350
def owner : Owner := ⟨.program ⟨257⟩, ⟨37785⟩⟩
def mergeEvent : Nat := 19350
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events075.exact19345RawTerms
def group : MergeGroup := .relation 19347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19347) (rhsResult := 19345)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩]⟩) (none) 19345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨37358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19350

namespace LeftMerge19351
def owner : Owner := ⟨.program ⟨257⟩, ⟨37785⟩⟩
def mergeEvent : Nat := 19351
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events075.exact19345RawTerms
def group : MergeGroup := .relation 19347
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19347) (rhsResult := 19345)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19346 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37782⟩⟩]⟩) (none) 19345) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19351

namespace LeftMerge19356
def owner : Owner := ⟨.program ⟨257⟩, ⟨38845⟩⟩
def mergeEvent : Nat := 19356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38377⟩⟩] } }
def leftRaw : List Term := Proof.Events075.exact19352RawTerms
def rightRaw : List Term := Proof.Events074.exact19166RawTerms
def group : MergeGroup := .operator 19352 19166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19352) (leftOrdinal := 2)
    (rightResult := 19166) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38377⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38377⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13751⟩⟩, ⟨.program ⟨257⟩, ⟨36906⟩⟩], [⟨.program ⟨257⟩, ⟨38377⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19356

namespace LeftMerge19357
def owner : Owner := ⟨.program ⟨257⟩, ⟨38845⟩⟩
def mergeEvent : Nat := 19357
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩] } }
def leftRaw : List Term := Proof.Events075.exact19352RawTerms
def rightRaw : List Term := Proof.Events074.exact19166RawTerms
def group : MergeGroup := .operator 19352 19166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19352) (leftOrdinal := 1)
    (rightResult := 19166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38843⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19357

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
