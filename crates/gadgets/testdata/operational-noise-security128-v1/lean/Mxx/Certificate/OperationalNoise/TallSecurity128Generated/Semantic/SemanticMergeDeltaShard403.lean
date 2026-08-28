import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge69074
def owner : Owner := ⟨.program ⟨257⟩, ⟨20297⟩⟩
def mergeEvent : Nat := 69074
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩] } }
def leftRaw : List Term := Proof.Events269.exact69065RawTerms
def rightRaw : List Term := Proof.Events269.exact69001RawTerms
def group : MergeGroup := .operator 69065 69001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69065) (leftOrdinal := 0)
    (rightResult := 69001) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20296⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69074

namespace LeftMerge69088
def owner : Owner := ⟨.program ⟨257⟩, ⟨19222⟩⟩
def mergeEvent : Nat := 69088
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events269.exact69082RawTerms
def group : MergeGroup := .operator 61370 69082
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 69082) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19219⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69088

namespace LeftMerge69167
def owner : Owner := ⟨.program ⟨257⟩, ⟨18443⟩⟩
def mergeEvent : Nat := 69167
def frameStart : Nat := 69137
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events270.exact69163RawTerms
def rightRaw : List Term := Proof.Events270.exact69160RawTerms
def group : MergeGroup := .operator 69163 69160
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69163) (leftOrdinal := 0)
    (rightResult := 69160) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12786⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69167

namespace LeftMerge69197
def owner : Owner := ⟨.program ⟨257⟩, ⟨20016⟩⟩
def mergeEvent : Nat := 69197
def frameStart : Nat := 69137
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events270.exact69193RawTerms
def rightRaw : List Term := Proof.Events270.exact69191RawTerms
def group : MergeGroup := .operator 69193 69191
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69193) (leftOrdinal := 0)
    (rightResult := 69191) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69197

namespace LeftMerge69220
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def mergeEvent : Nat := 69220
def frameStart : Nat := 69137
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events270.exact69216RawTerms
def rightRaw : List Term := Proof.Events270.exact69213RawTerms
def group : MergeGroup := .operator 69216 69213
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69216) (leftOrdinal := 0)
    (rightResult := 69213) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69220

namespace LeftMerge69229
def owner : Owner := ⟨.program ⟨257⟩, ⟨20299⟩⟩
def mergeEvent : Nat := 69229
def frameStart : Nat := 69137
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩] } }
def leftRaw : List Term := Proof.Events270.exact69225RawTerms
def rightRaw : List Term := Proof.Events270.exact69182RawTerms
def group : MergeGroup := .operator 69225 69182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69225) (leftOrdinal := 0)
    (rightResult := 69182) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20296⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69229

namespace LeftMerge69230
def owner : Owner := ⟨.program ⟨257⟩, ⟨20299⟩⟩
def mergeEvent : Nat := 69230
def frameStart : Nat := 69137
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩] } }
def leftRaw : List Term := Proof.Events270.exact69225RawTerms
def rightRaw : List Term := Proof.Events270.exact69182RawTerms
def group : MergeGroup := .operator 69225 69182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69225) (leftOrdinal := 1)
    (rightResult := 69182) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20296⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69230

namespace LeftMerge69232
def owner : Owner := ⟨.program ⟨257⟩, ⟨20299⟩⟩
def mergeEvent : Nat := 69232
def frameStart : Nat := 69137
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19751⟩⟩] } }
def rhsRaw : List Term := Proof.Events270.exact69179RawTerms
def group : MergeGroup := .relation 69231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69231) (rhsResult := 69179)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20296⟩⟩) ⟨19751⟩ 69179) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19751⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], [⟨.program ⟨257⟩, ⟨19751⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69232

namespace LeftMerge69240
def owner : Owner := ⟨.program ⟨257⟩, ⟨18646⟩⟩
def mergeEvent : Nat := 69240
def frameStart : Nat := 69137
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18644⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events270.exact69193RawTerms
def rightRaw : List Term := Proof.Events270.exact69236RawTerms
def group : MergeGroup := .operator 69193 69236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69193) (leftOrdinal := 0)
    (rightResult := 69236) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18644⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18644⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69240

namespace LeftMerge69257
def owner : Owner := ⟨.program ⟨257⟩, ⟨19222⟩⟩
def mergeEvent : Nat := 69257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events270.exact69254RawTerms
def group : MergeGroup := .relation 69256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69256) (rhsResult := 69254)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69255 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩]⟩) (none) 69254) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69257

namespace LeftMerge69258
def owner : Owner := ⟨.program ⟨257⟩, ⟨19222⟩⟩
def mergeEvent : Nat := 69258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩] } }
def rhsRaw : List Term := Proof.Events270.exact69254RawTerms
def group : MergeGroup := .relation 69256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69256) (rhsResult := 69254)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69255 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩]⟩) (none) 69254) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69258

namespace LeftMerge69259
def owner : Owner := ⟨.program ⟨257⟩, ⟨19222⟩⟩
def mergeEvent : Nat := 69259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19751⟩⟩] } }
def rhsRaw : List Term := Proof.Events270.exact69254RawTerms
def group : MergeGroup := .relation 69256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69256) (rhsResult := 69254)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69255 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩]⟩) (none) 69254) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19751⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], [⟨.program ⟨257⟩, ⟨19751⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69259

namespace LeftMerge69260
def owner : Owner := ⟨.program ⟨257⟩, ⟨19222⟩⟩
def mergeEvent : Nat := 69260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18644⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events270.exact69254RawTerms
def group : MergeGroup := .relation 69256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 69256) (rhsResult := 69254)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 69255 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19219⟩⟩]⟩) (none) 69254) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18644⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨18644⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69260

namespace LeftMerge69265
def owner : Owner := ⟨.program ⟨257⟩, ⟨20298⟩⟩
def mergeEvent : Nat := 69265
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19751⟩⟩] } }
def leftRaw : List Term := Proof.Events270.exact69261RawTerms
def rightRaw : List Term := Proof.Events269.exact69075RawTerms
def group : MergeGroup := .operator 69261 69075
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69261) (leftOrdinal := 2)
    (rightResult := 69075) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19751⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19751⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨12786⟩⟩, ⟨.program ⟨257⟩, ⟨18442⟩⟩], [⟨.program ⟨257⟩, ⟨19751⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge69265

namespace LeftMerge69266
def owner : Owner := ⟨.program ⟨257⟩, ⟨20298⟩⟩
def mergeEvent : Nat := 69266
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩] } }
def leftRaw : List Term := Proof.Events270.exact69261RawTerms
def rightRaw : List Term := Proof.Events269.exact69075RawTerms
def group : MergeGroup := .operator 69261 69075
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69261) (leftOrdinal := 1)
    (rightResult := 69075) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69266

namespace LeftMerge69274
def owner : Owner := ⟨.program ⟨257⟩, ⟨20871⟩⟩
def mergeEvent : Nat := 69274
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20869⟩⟩] } }
def leftRaw : List Term := Proof.Events270.exact69268RawTerms
def rightRaw : List Term := Proof.Events269.exact68991RawTerms
def group : MergeGroup := .operator 69268 68991
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 69268) (leftOrdinal := 0)
    (rightResult := 68991) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20869⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge69274

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
