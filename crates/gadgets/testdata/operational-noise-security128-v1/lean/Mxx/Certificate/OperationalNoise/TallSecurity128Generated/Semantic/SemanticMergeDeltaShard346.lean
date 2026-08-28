import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge60322
def owner : Owner := ⟨.program ⟨257⟩, ⟨22835⟩⟩
def mergeEvent : Nat := 60322
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩] } }
def rhsRaw : List Term := Proof.Events235.exact60318RawTerms
def group : MergeGroup := .relation 60320
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60320) (rhsResult := 60318)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60319 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩) (none) 60318) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60322

namespace LeftMerge60323
def owner : Owner := ⟨.program ⟨257⟩, ⟨22835⟩⟩
def mergeEvent : Nat := 60323
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23152⟩⟩] } }
def rhsRaw : List Term := Proof.Events235.exact60318RawTerms
def group : MergeGroup := .relation 60320
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60320) (rhsResult := 60318)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60319 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩) (none) 60318) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23152⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60323

namespace LeftMerge60324
def owner : Owner := ⟨.program ⟨257⟩, ⟨22835⟩⟩
def mergeEvent : Nat := 60324
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events235.exact60318RawTerms
def group : MergeGroup := .relation 60320
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60320) (rhsResult := 60318)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 60319 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22832⟩⟩]⟩) (none) 60318) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60324

namespace LeftMerge60329
def owner : Owner := ⟨.program ⟨257⟩, ⟨24116⟩⟩
def mergeEvent : Nat := 60329
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩] } }
def leftRaw : List Term := Proof.Events235.exact60325RawTerms
def rightRaw : List Term := Proof.Events234.exact60147RawTerms
def group : MergeGroup := .operator 60325 60147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60325) (leftOrdinal := 0)
    (rightResult := 60147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60329

namespace LeftMerge60330
def owner : Owner := ⟨.program ⟨257⟩, ⟨24116⟩⟩
def mergeEvent : Nat := 60330
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23152⟩⟩] } }
def leftRaw : List Term := Proof.Events235.exact60325RawTerms
def rightRaw : List Term := Proof.Events234.exact60147RawTerms
def group : MergeGroup := .operator 60325 60147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60325) (leftOrdinal := 2)
    (rightResult := 60147) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23152⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23152⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨21872⟩⟩], [⟨.program ⟨257⟩, ⟨23152⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60330

namespace LeftMerge60338
def owner : Owner := ⟨.program ⟨257⟩, ⟨24117⟩⟩
def mergeEvent : Nat := 60338
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩] } }
def leftRaw : List Term := Proof.Events235.exact60332RawTerms
def rightRaw : List Term := Proof.Events061.exact15842RawTerms
def group : MergeGroup := .operator 60332 15842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60332) (leftOrdinal := 0)
    (rightResult := 15842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7201⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7155⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7201⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60338

namespace LeftMerge60339
def owner : Owner := ⟨.program ⟨257⟩, ⟨24117⟩⟩
def mergeEvent : Nat := 60339
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩] } }
def leftRaw : List Term := Proof.Events235.exact60332RawTerms
def rightRaw : List Term := Proof.Events061.exact15842RawTerms
def group : MergeGroup := .operator 60332 15842
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60332) (leftOrdinal := 1)
    (rightResult := 15842) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7155⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60339

namespace LeftMerge60341
def owner : Owner := ⟨.program ⟨257⟩, ⟨24117⟩⟩
def mergeEvent : Nat := 60341
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15835RawTerms
def group : MergeGroup := .relation 60340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60340) (rhsResult := 15835)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7155⟩⟩) ⟨7043⟩ 15835) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨22233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60341

namespace LeftMerge60355
def owner : Owner := ⟨.program ⟨257⟩, ⟨20895⟩⟩
def mergeEvent : Nat := 60355
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54643RawTerms
def rightRaw : List Term := Proof.Events235.exact60349RawTerms
def group : MergeGroup := .operator 54643 60349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54643) (leftOrdinal := 0)
    (rightResult := 60349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20893⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60355

namespace LeftMerge60356
def owner : Owner := ⟨.program ⟨257⟩, ⟨20895⟩⟩
def mergeEvent : Nat := 60356
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54643RawTerms
def rightRaw : List Term := Proof.Events235.exact60349RawTerms
def group : MergeGroup := .operator 54643 60349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54643) (leftOrdinal := 1)
    (rightResult := 60349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20893⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60356

namespace LeftMerge60358
def owner : Owner := ⟨.program ⟨257⟩, ⟨20895⟩⟩
def mergeEvent : Nat := 60358
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19932⟩⟩] } }
def rhsRaw : List Term := Proof.Events235.exact60346RawTerms
def group : MergeGroup := .relation 60357
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60357) (rhsResult := 60346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20893⟩⟩) ⟨19932⟩ 60346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19932⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨19932⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60358

namespace LeftMerge60372
def owner : Owner := ⟨.program ⟨257⟩, ⟨19615⟩⟩
def mergeEvent : Nat := 60372
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19612⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events235.exact60366RawTerms
def group : MergeGroup := .operator 46745 60366
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 60366) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19612⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60372

namespace LeftMerge60493
def owner : Owner := ⟨.program ⟨257⟩, ⟨20100⟩⟩
def mergeEvent : Nat := 60493
def frameStart : Nat := 60427
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events236.exact60489RawTerms
def rightRaw : List Term := Proof.Events236.exact60487RawTerms
def group : MergeGroup := .operator 60489 60487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60489) (leftOrdinal := 0)
    (rightResult := 60487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60493

namespace LeftMerge60505
def owner : Owner := ⟨.program ⟨257⟩, ⟨20894⟩⟩
def mergeEvent : Nat := 60505
def frameStart : Nat := 60427
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩] } }
def leftRaw : List Term := Proof.Events236.exact60501RawTerms
def rightRaw : List Term := Proof.Events236.exact60478RawTerms
def group : MergeGroup := .operator 60501 60478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60501) (leftOrdinal := 0)
    (rightResult := 60478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20893⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge60505

namespace LeftMerge60506
def owner : Owner := ⟨.program ⟨257⟩, ⟨20894⟩⟩
def mergeEvent : Nat := 60506
def frameStart : Nat := 60427
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩] } }
def leftRaw : List Term := Proof.Events236.exact60501RawTerms
def rightRaw : List Term := Proof.Events236.exact60478RawTerms
def group : MergeGroup := .operator 60501 60478
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 60501) (leftOrdinal := 1)
    (rightResult := 60478) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20893⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60506

namespace LeftMerge60508
def owner : Owner := ⟨.program ⟨257⟩, ⟨20894⟩⟩
def mergeEvent : Nat := 60508
def frameStart : Nat := 60427
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19932⟩⟩] } }
def rhsRaw : List Term := Proof.Events236.exact60475RawTerms
def group : MergeGroup := .relation 60507
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 60507) (rhsResult := 60475)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20893⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20893⟩⟩) ⟨19932⟩ 60475) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19932⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨19932⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge60508

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
