import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge43169
def owner : Owner := ⟨.program ⟨257⟩, ⟨42212⟩⟩
def mergeEvent : Nat := 43169
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43163RawTerms
def rightRaw : List Term := Proof.Events060.exact15602RawTerms
def group : MergeGroup := .operator 43163 15602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43163) (leftOrdinal := 0)
    (rightResult := 15602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7225⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7225⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43169

namespace LeftMerge43170
def owner : Owner := ⟨.program ⟨257⟩, ⟨42212⟩⟩
def mergeEvent : Nat := 43170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩] } }
def leftRaw : List Term := Proof.Events168.exact43163RawTerms
def rightRaw : List Term := Proof.Events060.exact15602RawTerms
def group : MergeGroup := .operator 43163 15602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43163) (leftOrdinal := 1)
    (rightResult := 15602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43170

namespace LeftMerge43172
def owner : Owner := ⟨.program ⟨257⟩, ⟨42212⟩⟩
def mergeEvent : Nat := 43172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15595RawTerms
def group : MergeGroup := .relation 43171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43171) (rhsResult := 15595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7159⟩⟩) ⟨7045⟩ 15595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨40439⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43172

namespace LeftMerge43186
def owner : Owner := ⟨.program ⟨257⟩, ⟨39530⟩⟩
def mergeEvent : Nat := 43186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34234RawTerms
def rightRaw : List Term := Proof.Events168.exact43180RawTerms
def group : MergeGroup := .operator 34234 43180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34234) (leftOrdinal := 0)
    (rightResult := 43180) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39528⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43186

namespace LeftMerge43187
def owner : Owner := ⟨.program ⟨257⟩, ⟨39530⟩⟩
def mergeEvent : Nat := 43187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩] } }
def leftRaw : List Term := Proof.Events133.exact34234RawTerms
def rightRaw : List Term := Proof.Events168.exact43180RawTerms
def group : MergeGroup := .operator 34234 43180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 34234) (leftOrdinal := 1)
    (rightResult := 43180) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39528⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43187

namespace LeftMerge43189
def owner : Owner := ⟨.program ⟨257⟩, ⟨39530⟩⟩
def mergeEvent : Nat := 43189
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38661⟩⟩] } }
def rhsRaw : List Term := Proof.Events168.exact43177RawTerms
def group : MergeGroup := .relation 43188
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43188) (rhsResult := 43177)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39528⟩⟩) ⟨38661⟩ 43177) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38661⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43189

namespace LeftMerge43203
def owner : Owner := ⟨.program ⟨257⟩, ⟨38355⟩⟩
def mergeEvent : Nat := 43203
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events168.exact43197RawTerms
def group : MergeGroup := .operator 32120 43197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 43197) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38352⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43203

namespace LeftMerge43324
def owner : Owner := ⟨.program ⟨257⟩, ⟨38824⟩⟩
def mergeEvent : Nat := 43324
def frameStart : Nat := 43258
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43320RawTerms
def rightRaw : List Term := Proof.Events169.exact43318RawTerms
def group : MergeGroup := .operator 43320 43318
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43320) (leftOrdinal := 0)
    (rightResult := 43318) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43324

namespace LeftMerge43336
def owner : Owner := ⟨.program ⟨257⟩, ⟨39529⟩⟩
def mergeEvent : Nat := 43336
def frameStart : Nat := 43258
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43332RawTerms
def rightRaw : List Term := Proof.Events169.exact43309RawTerms
def group : MergeGroup := .operator 43332 43309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43332) (leftOrdinal := 0)
    (rightResult := 43309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39528⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43336

namespace LeftMerge43337
def owner : Owner := ⟨.program ⟨257⟩, ⟨39529⟩⟩
def mergeEvent : Nat := 43337
def frameStart : Nat := 43258
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43332RawTerms
def rightRaw : List Term := Proof.Events169.exact43309RawTerms
def group : MergeGroup := .operator 43332 43309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43332) (leftOrdinal := 1)
    (rightResult := 43309) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39528⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43337

namespace LeftMerge43339
def owner : Owner := ⟨.program ⟨257⟩, ⟨39529⟩⟩
def mergeEvent : Nat := 43339
def frameStart : Nat := 43258
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38661⟩⟩] } }
def rhsRaw : List Term := Proof.Events169.exact43306RawTerms
def group : MergeGroup := .relation 43338
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43338) (rhsResult := 43306)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39528⟩⟩) ⟨38661⟩ 43306) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38661⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43339

namespace LeftMerge43347
def owner : Owner := ⟨.program ⟨257⟩, ⟨37758⟩⟩
def mergeEvent : Nat := 43347
def frameStart : Nat := 43258
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events169.exact43320RawTerms
def rightRaw : List Term := Proof.Events169.exact43343RawTerms
def group : MergeGroup := .operator 43320 43343
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 43320) (leftOrdinal := 0)
    (rightResult := 43343) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37756⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43347

namespace LeftMerge43364
def owner : Owner := ⟨.program ⟨257⟩, ⟨38355⟩⟩
def mergeEvent : Nat := 43364
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩] } }
def rhsRaw : List Term := Proof.Events169.exact43361RawTerms
def group : MergeGroup := .relation 43363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43363) (rhsResult := 43361)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩) (none) 43361) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43364

namespace LeftMerge43365
def owner : Owner := ⟨.program ⟨257⟩, ⟨38355⟩⟩
def mergeEvent : Nat := 43365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩] } }
def rhsRaw : List Term := Proof.Events169.exact43361RawTerms
def group : MergeGroup := .relation 43363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43363) (rhsResult := 43361)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩) (none) 43361) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39528⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43365

namespace LeftMerge43366
def owner : Owner := ⟨.program ⟨257⟩, ⟨38355⟩⟩
def mergeEvent : Nat := 43366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38661⟩⟩] } }
def rhsRaw : List Term := Proof.Events169.exact43361RawTerms
def group : MergeGroup := .relation 43363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43363) (rhsResult := 43361)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩) (none) 43361) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37500⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38661⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37500⟩⟩], [⟨.program ⟨257⟩, ⟨38661⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge43366

namespace LeftMerge43367
def owner : Owner := ⟨.program ⟨257⟩, ⟨38355⟩⟩
def mergeEvent : Nat := 43367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events169.exact43361RawTerms
def group : MergeGroup := .relation 43363
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 43363) (rhsResult := 43361)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 43362 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38352⟩⟩]⟩) (none) 43361) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37756⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨37756⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge43367

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
