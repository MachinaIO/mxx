import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge254248
def owner : Owner := ⟨.program ⟨257⟩, ⟨36505⟩⟩
def mergeEvent : Nat := 254248
def frameStart : Nat := 254169
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩] } }
def leftRaw : List Term := Proof.Events993.exact254243RawTerms
def rightRaw : List Term := Proof.Events993.exact254220RawTerms
def group : MergeGroup := .operator 254243 254220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254243) (leftOrdinal := 1)
    (rightResult := 254220) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36504⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254248

namespace LeftMerge254250
def owner : Owner := ⟨.program ⟨257⟩, ⟨36505⟩⟩
def mergeEvent : Nat := 254250
def frameStart : Nat := 254169
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35856⟩⟩] } }
def rhsRaw : List Term := Proof.Events993.exact254217RawTerms
def group : MergeGroup := .relation 254249
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254249) (rhsResult := 254217)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36504⟩⟩) ⟨35856⟩ 254217) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35856⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34708⟩⟩], [⟨.program ⟨257⟩, ⟨35856⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254250

namespace LeftMerge254258
def owner : Owner := ⟨.program ⟨257⟩, ⟨34899⟩⟩
def mergeEvent : Nat := 254258
def frameStart : Nat := 254169
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events993.exact254231RawTerms
def rightRaw : List Term := Proof.Events993.exact254254RawTerms
def group : MergeGroup := .operator 254231 254254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254231) (leftOrdinal := 0)
    (rightResult := 254254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34898⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254258

namespace LeftMerge254275
def owner : Owner := ⟨.program ⟨257⟩, ⟨35399⟩⟩
def mergeEvent : Nat := 254275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }
def rhsRaw : List Term := Proof.Events993.exact254272RawTerms
def group : MergeGroup := .relation 254274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254274) (rhsResult := 254272)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35396⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 254273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35396⟩⟩]⟩) (none) 254272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254275

namespace LeftMerge254276
def owner : Owner := ⟨.program ⟨257⟩, ⟨35399⟩⟩
def mergeEvent : Nat := 254276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩] } }
def rhsRaw : List Term := Proof.Events993.exact254272RawTerms
def group : MergeGroup := .relation 254274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254274) (rhsResult := 254272)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35396⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 254273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35396⟩⟩]⟩) (none) 254272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254276

namespace LeftMerge254277
def owner : Owner := ⟨.program ⟨257⟩, ⟨35399⟩⟩
def mergeEvent : Nat := 254277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨34708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35856⟩⟩] } }
def rhsRaw : List Term := Proof.Events993.exact254272RawTerms
def group : MergeGroup := .relation 254274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254274) (rhsResult := 254272)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35396⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 254273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35396⟩⟩]⟩) (none) 254272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35856⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨34708⟩⟩], [⟨.program ⟨257⟩, ⟨35856⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254277

namespace LeftMerge254278
def owner : Owner := ⟨.program ⟨257⟩, ⟨35399⟩⟩
def mergeEvent : Nat := 254278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨34898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events993.exact254272RawTerms
def group : MergeGroup := .relation 254274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254274) (rhsResult := 254272)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35396⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 254273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35396⟩⟩]⟩) (none) 254272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨34898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254278

namespace LeftMerge254283
def owner : Owner := ⟨.program ⟨257⟩, ⟨36507⟩⟩
def mergeEvent : Nat := 254283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩] } }
def leftRaw : List Term := Proof.Events993.exact254279RawTerms
def rightRaw : List Term := Proof.Events992.exact254101RawTerms
def group : MergeGroup := .operator 254279 254101
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254279) (leftOrdinal := 0)
    (rightResult := 254101) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36504⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254283

namespace LeftMerge254284
def owner : Owner := ⟨.program ⟨257⟩, ⟨36507⟩⟩
def mergeEvent : Nat := 254284
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨34708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35856⟩⟩] } }
def leftRaw : List Term := Proof.Events993.exact254279RawTerms
def rightRaw : List Term := Proof.Events992.exact254101RawTerms
def group : MergeGroup := .operator 254279 254101
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254279) (leftOrdinal := 2)
    (rightResult := 254101) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨34708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35856⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨34708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35856⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨34708⟩⟩], [⟨.program ⟨257⟩, ⟨35856⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254284

namespace LeftMerge254310
def owner : Owner := ⟨.program ⟨257⟩, ⟨28657⟩⟩
def mergeEvent : Nat := 254310
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events047.exact12200RawTerms
def rightRaw : List Term := Proof.Events982.exact251403RawTerms
def group : MergeGroup := .operator 12200 251403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12200) (leftOrdinal := 0)
    (rightResult := 251403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28654⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254310

namespace LeftMerge254315
def owner : Owner := ⟨.program ⟨257⟩, ⟨8015⟩⟩
def mergeEvent : Nat := 254315
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251273RawTerms
def rightRaw : List Term := Proof.Events078.exact20086RawTerms
def group : MergeGroup := .operator 251273 20086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251273) (leftOrdinal := 0)
    (rightResult := 20086) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254315

namespace LeftMerge254332
def owner : Owner := ⟨.program ⟨257⟩, ⟨28660⟩⟩
def mergeEvent : Nat := 254332
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events993.exact254326RawTerms
def rightRaw : List Term := Proof.Events047.exact12203RawTerms
def group : MergeGroup := .operator 254326 12203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254326) (leftOrdinal := 1)
    (rightResult := 12203) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13206⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254332

namespace LeftMerge254333
def owner : Owner := ⟨.program ⟨257⟩, ⟨28660⟩⟩
def mergeEvent : Nat := 254333
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events993.exact254326RawTerms
def rightRaw : List Term := Proof.Events047.exact12203RawTerms
def group : MergeGroup := .operator 254326 12203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254326) (leftOrdinal := 0)
    (rightResult := 12203) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13206⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254333

namespace LeftMerge254338
def owner : Owner := ⟨.program ⟨257⟩, ⟨13207⟩⟩
def mergeEvent : Nat := 254338
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events047.exact12203RawTerms
def rightRaw : List Term := Proof.Events982.exact251403RawTerms
def group : MergeGroup := .operator 12203 251403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 12203) (leftOrdinal := 0)
    (rightResult := 251403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13206⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254338

namespace LeftMerge254343
def owner : Owner := ⟨.program ⟨257⟩, ⟨8032⟩⟩
def mergeEvent : Nat := 254343
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }
def leftRaw : List Term := Proof.Events981.exact251273RawTerms
def rightRaw : List Term := Proof.Events078.exact20127RawTerms
def group : MergeGroup := .operator 251273 20127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251273) (leftOrdinal := 0)
    (rightResult := 20127) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254343

namespace LeftMerge254360
def owner : Owner := ⟨.program ⟨257⟩, ⟨13210⟩⟩
def mergeEvent : Nat := 254360
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events993.exact254354RawTerms
def rightRaw : List Term := Proof.Events078.exact20116RawTerms
def group : MergeGroup := .operator 254354 20116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254354) (leftOrdinal := 1)
    (rightResult := 20116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254360

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
