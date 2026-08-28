import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge66166
def owner : Owner := ⟨.program ⟨257⟩, ⟨59681⟩⟩
def mergeEvent : Nat := 66166
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events258.exact66157RawTerms
def rightRaw : List Term := Proof.Events086.exact22120RawTerms
def group : MergeGroup := .operator 66157 22120
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66157) (leftOrdinal := 0)
    (rightResult := 22120) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66166

namespace LeftMerge66171
def owner : Owner := ⟨.program ⟨257⟩, ⟨59682⟩⟩
def mergeEvent : Nat := 66171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events258.exact66167RawTerms
def rightRaw : List Term := Proof.Events258.exact66137RawTerms
def group : MergeGroup := .operator 66167 66137
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66167) (leftOrdinal := 1)
    (rightResult := 66137) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66171

namespace LeftMerge66179
def owner : Owner := ⟨.program ⟨257⟩, ⟨61537⟩⟩
def mergeEvent : Nat := 66179
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩] } }
def leftRaw : List Term := Proof.Events258.exact66173RawTerms
def rightRaw : List Term := Proof.Events258.exact66109RawTerms
def group : MergeGroup := .operator 66173 66109
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66173) (leftOrdinal := 1)
    (rightResult := 66109) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66179

namespace LeftMerge66181
def owner : Owner := ⟨.program ⟨257⟩, ⟨61537⟩⟩
def mergeEvent : Nat := 66181
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60991⟩⟩] } }
def rhsRaw : List Term := Proof.Events258.exact66106RawTerms
def group : MergeGroup := .relation 66180
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66180) (rhsResult := 66106)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61536⟩⟩) ⟨60991⟩ 66106) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60991⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨60991⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66181

namespace LeftMerge66182
def owner : Owner := ⟨.program ⟨257⟩, ⟨61537⟩⟩
def mergeEvent : Nat := 66182
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩] } }
def leftRaw : List Term := Proof.Events258.exact66173RawTerms
def rightRaw : List Term := Proof.Events258.exact66109RawTerms
def group : MergeGroup := .operator 66173 66109
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66173) (leftOrdinal := 0)
    (rightResult := 66109) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66182

namespace LeftMerge66196
def owner : Owner := ⟨.program ⟨257⟩, ⟨60462⟩⟩
def mergeEvent : Nat := 66196
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events258.exact66190RawTerms
def group : MergeGroup := .operator 61370 66190
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 66190) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66196

namespace LeftMerge66275
def owner : Owner := ⟨.program ⟨257⟩, ⟨59675⟩⟩
def mergeEvent : Nat := 66275
def frameStart : Nat := 66245
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events258.exact66271RawTerms
def rightRaw : List Term := Proof.Events258.exact66268RawTerms
def group : MergeGroup := .operator 66271 66268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66271) (leftOrdinal := 0)
    (rightResult := 66268) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25334⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66275

namespace LeftMerge66305
def owner : Owner := ⟨.program ⟨257⟩, ⟨61256⟩⟩
def mergeEvent : Nat := 66305
def frameStart : Nat := 66245
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events258.exact66301RawTerms
def rightRaw : List Term := Proof.Events258.exact66299RawTerms
def group : MergeGroup := .operator 66301 66299
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66301) (leftOrdinal := 0)
    (rightResult := 66299) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66305

namespace LeftMerge66328
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def mergeEvent : Nat := 66328
def frameStart : Nat := 66245
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66324RawTerms
def rightRaw : List Term := Proof.Events259.exact66321RawTerms
def group : MergeGroup := .operator 66324 66321
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66324) (leftOrdinal := 0)
    (rightResult := 66321) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9535⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66328

namespace LeftMerge66337
def owner : Owner := ⟨.program ⟨257⟩, ⟨61539⟩⟩
def mergeEvent : Nat := 66337
def frameStart : Nat := 66245
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66333RawTerms
def rightRaw : List Term := Proof.Events258.exact66290RawTerms
def group : MergeGroup := .operator 66333 66290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66333) (leftOrdinal := 0)
    (rightResult := 66290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61536⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66337

namespace LeftMerge66338
def owner : Owner := ⟨.program ⟨257⟩, ⟨61539⟩⟩
def mergeEvent : Nat := 66338
def frameStart : Nat := 66245
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩] } }
def leftRaw : List Term := Proof.Events259.exact66333RawTerms
def rightRaw : List Term := Proof.Events258.exact66290RawTerms
def group : MergeGroup := .operator 66333 66290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66333) (leftOrdinal := 1)
    (rightResult := 66290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61536⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66338

namespace LeftMerge66340
def owner : Owner := ⟨.program ⟨257⟩, ⟨61539⟩⟩
def mergeEvent : Nat := 66340
def frameStart : Nat := 66245
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60991⟩⟩] } }
def rhsRaw : List Term := Proof.Events258.exact66287RawTerms
def group : MergeGroup := .relation 66339
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66339) (rhsResult := 66287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61536⟩⟩) ⟨60991⟩ 66287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60991⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨60991⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66340

namespace LeftMerge66348
def owner : Owner := ⟨.program ⟨257⟩, ⟨59886⟩⟩
def mergeEvent : Nat := 66348
def frameStart : Nat := 66245
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events258.exact66301RawTerms
def rightRaw : List Term := Proof.Events259.exact66344RawTerms
def group : MergeGroup := .operator 66301 66344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 66301) (leftOrdinal := 0)
    (rightResult := 66344) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59884⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59884⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66348

namespace LeftMerge66365
def owner : Owner := ⟨.program ⟨257⟩, ⟨60462⟩⟩
def mergeEvent : Nat := 66365
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66362RawTerms
def group : MergeGroup := .relation 66364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66364) (rhsResult := 66362)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩]⟩) (none) 66362) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66365

namespace LeftMerge66366
def owner : Owner := ⟨.program ⟨257⟩, ⟨60462⟩⟩
def mergeEvent : Nat := 66366
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66362RawTerms
def group : MergeGroup := .relation 66364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66364) (rhsResult := 66362)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩]⟩) (none) 66362) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7291⟩⟩, ⟨.program ⟨257⟩, ⟨9535⟩⟩, ⟨.program ⟨257⟩, ⟨61536⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge66366

namespace LeftMerge66367
def owner : Owner := ⟨.program ⟨257⟩, ⟨60462⟩⟩
def mergeEvent : Nat := 66367
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60991⟩⟩] } }
def rhsRaw : List Term := Proof.Events259.exact66362RawTerms
def group : MergeGroup := .relation 66364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 66364) (rhsResult := 66362)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 66363 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60459⟩⟩]⟩) (none) 66362) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨60991⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨25334⟩⟩, ⟨.program ⟨257⟩, ⟨59674⟩⟩], [⟨.program ⟨257⟩, ⟨60991⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge66367

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
