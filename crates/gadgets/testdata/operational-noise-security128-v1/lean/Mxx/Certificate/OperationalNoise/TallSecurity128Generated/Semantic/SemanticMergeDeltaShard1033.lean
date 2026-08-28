import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge169197
def owner : Owner := ⟨.program ⟨257⟩, ⟨58526⟩⟩
def mergeEvent : Nat := 169197
def frameStart : Nat := 169102
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57993⟩⟩] } }
def rhsRaw : List Term := Proof.Events660.exact169144RawTerms
def group : MergeGroup := .relation 169196
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169196) (rhsResult := 169144)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58523⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58523⟩⟩) ⟨57993⟩ 169144) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57993⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], [⟨.program ⟨257⟩, ⟨57993⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169197

namespace LeftMerge169205
def owner : Owner := ⟨.program ⟨257⟩, ⟨56882⟩⟩
def mergeEvent : Nat := 169205
def frameStart : Nat := 169102
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events660.exact169158RawTerms
def rightRaw : List Term := Proof.Events660.exact169201RawTerms
def group : MergeGroup := .operator 169158 169201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169158) (leftOrdinal := 0)
    (rightResult := 169201) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169205

namespace LeftMerge169222
def owner : Owner := ⟨.program ⟨257⟩, ⟨57452⟩⟩
def mergeEvent : Nat := 169222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }
def rhsRaw : List Term := Proof.Events661.exact169219RawTerms
def group : MergeGroup := .relation 169221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169221) (rhsResult := 169219)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 169220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57449⟩⟩]⟩) (none) 169219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169222

namespace LeftMerge169223
def owner : Owner := ⟨.program ⟨257⟩, ⟨57452⟩⟩
def mergeEvent : Nat := 169223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58523⟩⟩] } }
def rhsRaw : List Term := Proof.Events661.exact169219RawTerms
def group : MergeGroup := .relation 169221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169221) (rhsResult := 169219)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 169220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57449⟩⟩]⟩) (none) 169219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58523⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58523⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169223

namespace LeftMerge169224
def owner : Owner := ⟨.program ⟨257⟩, ⟨57452⟩⟩
def mergeEvent : Nat := 169224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57993⟩⟩] } }
def rhsRaw : List Term := Proof.Events661.exact169219RawTerms
def group : MergeGroup := .relation 169221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169221) (rhsResult := 169219)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 169220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57449⟩⟩]⟩) (none) 169219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57993⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], [⟨.program ⟨257⟩, ⟨57993⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169224

namespace LeftMerge169225
def owner : Owner := ⟨.program ⟨257⟩, ⟨57452⟩⟩
def mergeEvent : Nat := 169225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events661.exact169219RawTerms
def group : MergeGroup := .relation 169221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169221) (rhsResult := 169219)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57449⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 169220 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57449⟩⟩]⟩) (none) 169219) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169225

namespace LeftMerge169230
def owner : Owner := ⟨.program ⟨257⟩, ⟨58525⟩⟩
def mergeEvent : Nat := 169230
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57993⟩⟩] } }
def leftRaw : List Term := Proof.Events661.exact169226RawTerms
def rightRaw : List Term := Proof.Events660.exact169040RawTerms
def group : MergeGroup := .operator 169226 169040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169226) (leftOrdinal := 2)
    (rightResult := 169040) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57993⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57993⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨25058⟩⟩, ⟨.program ⟨257⟩, ⟨56613⟩⟩], [⟨.program ⟨257⟩, ⟨57993⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169230

namespace LeftMerge169231
def owner : Owner := ⟨.program ⟨257⟩, ⟨58525⟩⟩
def mergeEvent : Nat := 169231
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58523⟩⟩] } }
def leftRaw : List Term := Proof.Events661.exact169226RawTerms
def rightRaw : List Term := Proof.Events660.exact169040RawTerms
def group : MergeGroup := .operator 169226 169040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169226) (leftOrdinal := 1)
    (rightResult := 169040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58523⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58523⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58523⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169231

namespace LeftMerge169239
def owner : Owner := ⟨.program ⟨257⟩, ⟨59038⟩⟩
def mergeEvent : Nat := 169239
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩] } }
def leftRaw : List Term := Proof.Events661.exact169233RawTerms
def rightRaw : List Term := Proof.Events659.exact168956RawTerms
def group : MergeGroup := .operator 169233 168956
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169233) (leftOrdinal := 0)
    (rightResult := 168956) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59036⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169239

namespace LeftMerge169240
def owner : Owner := ⟨.program ⟨257⟩, ⟨59038⟩⟩
def mergeEvent : Nat := 169240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩] } }
def leftRaw : List Term := Proof.Events661.exact169233RawTerms
def rightRaw : List Term := Proof.Events659.exact168956RawTerms
def group : MergeGroup := .operator 169233 168956
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169233) (leftOrdinal := 1)
    (rightResult := 168956) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59036⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169240

namespace LeftMerge169242
def owner : Owner := ⟨.program ⟨257⟩, ⟨59038⟩⟩
def mergeEvent : Nat := 169242
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58157⟩⟩] } }
def rhsRaw : List Term := Proof.Events659.exact168953RawTerms
def group : MergeGroup := .relation 169241
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169241) (rhsResult := 168953)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59036⟩⟩) ⟨58157⟩ 168953) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58157⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨58157⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169242

namespace LeftMerge169256
def owner : Owner := ⟨.program ⟨257⟩, ⟨57799⟩⟩
def mergeEvent : Nat := 169256
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57796⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events661.exact169250RawTerms
def group : MergeGroup := .operator 163745 169250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 169250) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57796⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57796⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169256

namespace LeftMerge169377
def owner : Owner := ⟨.program ⟨257⟩, ⟨58344⟩⟩
def mergeEvent : Nat := 169377
def frameStart : Nat := 169311
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events661.exact169373RawTerms
def rightRaw : List Term := Proof.Events661.exact169371RawTerms
def group : MergeGroup := .operator 169373 169371
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169373) (leftOrdinal := 0)
    (rightResult := 169371) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169377

namespace LeftMerge169389
def owner : Owner := ⟨.program ⟨257⟩, ⟨59037⟩⟩
def mergeEvent : Nat := 169389
def frameStart : Nat := 169311
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩] } }
def leftRaw : List Term := Proof.Events661.exact169385RawTerms
def rightRaw : List Term := Proof.Events661.exact169362RawTerms
def group : MergeGroup := .operator 169385 169362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169385) (leftOrdinal := 0)
    (rightResult := 169362) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59036⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge169389

namespace LeftMerge169390
def owner : Owner := ⟨.program ⟨257⟩, ⟨59037⟩⟩
def mergeEvent : Nat := 169390
def frameStart : Nat := 169311
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩] } }
def leftRaw : List Term := Proof.Events661.exact169385RawTerms
def rightRaw : List Term := Proof.Events661.exact169362RawTerms
def group : MergeGroup := .operator 169385 169362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 169385) (leftOrdinal := 1)
    (rightResult := 169362) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨59036⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169390

namespace LeftMerge169392
def owner : Owner := ⟨.program ⟨257⟩, ⟨59037⟩⟩
def mergeEvent : Nat := 169392
def frameStart : Nat := 169311
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58157⟩⟩] } }
def rhsRaw : List Term := Proof.Events661.exact169359RawTerms
def group : MergeGroup := .relation 169391
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 169391) (rhsResult := 169359)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨59036⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨59036⟩⟩) ⟨58157⟩ 169359) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58157⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56880⟩⟩], [⟨.program ⟨257⟩, ⟨58157⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge169392

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
