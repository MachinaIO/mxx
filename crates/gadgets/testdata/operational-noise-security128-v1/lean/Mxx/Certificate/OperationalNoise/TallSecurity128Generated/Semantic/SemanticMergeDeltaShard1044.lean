import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge171167
def owner : Owner := ⟨.program ⟨257⟩, ⟨23998⟩⟩
def mergeEvent : Nat := 171167
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171161RawTerms
def rightRaw : List Term := Proof.Events667.exact170884RawTerms
def group : MergeGroup := .operator 171161 170884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171161) (leftOrdinal := 0)
    (rightResult := 170884) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23996⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171167

namespace LeftMerge171168
def owner : Owner := ⟨.program ⟨257⟩, ⟨23998⟩⟩
def mergeEvent : Nat := 171168
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171161RawTerms
def rightRaw : List Term := Proof.Events667.exact170884RawTerms
def group : MergeGroup := .operator 171161 170884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171161) (leftOrdinal := 1)
    (rightResult := 170884) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23996⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171168

namespace LeftMerge171170
def owner : Owner := ⟨.program ⟨257⟩, ⟨23998⟩⟩
def mergeEvent : Nat := 171170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23117⟩⟩] } }
def rhsRaw : List Term := Proof.Events667.exact170881RawTerms
def group : MergeGroup := .relation 171169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171169) (rhsResult := 170881)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23996⟩⟩) ⟨23117⟩ 170881) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23117⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23117⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171170

namespace LeftMerge171184
def owner : Owner := ⟨.program ⟨257⟩, ⟨22759⟩⟩
def mergeEvent : Nat := 171184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events668.exact171178RawTerms
def group : MergeGroup := .operator 163745 171178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 171178) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22756⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171184

namespace LeftMerge171305
def owner : Owner := ⟨.program ⟨257⟩, ⟨23304⟩⟩
def mergeEvent : Nat := 171305
def frameStart : Nat := 171239
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events669.exact171301RawTerms
def rightRaw : List Term := Proof.Events669.exact171299RawTerms
def group : MergeGroup := .operator 171301 171299
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171301) (leftOrdinal := 0)
    (rightResult := 171299) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171305

namespace LeftMerge171317
def owner : Owner := ⟨.program ⟨257⟩, ⟨23997⟩⟩
def mergeEvent : Nat := 171317
def frameStart : Nat := 171239
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩] } }
def leftRaw : List Term := Proof.Events669.exact171313RawTerms
def rightRaw : List Term := Proof.Events669.exact171290RawTerms
def group : MergeGroup := .operator 171313 171290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171313) (leftOrdinal := 0)
    (rightResult := 171290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23996⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171317

namespace LeftMerge171318
def owner : Owner := ⟨.program ⟨257⟩, ⟨23997⟩⟩
def mergeEvent : Nat := 171318
def frameStart : Nat := 171239
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩] } }
def leftRaw : List Term := Proof.Events669.exact171313RawTerms
def rightRaw : List Term := Proof.Events669.exact171290RawTerms
def group : MergeGroup := .operator 171313 171290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171313) (leftOrdinal := 1)
    (rightResult := 171290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23996⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171318

namespace LeftMerge171320
def owner : Owner := ⟨.program ⟨257⟩, ⟨23997⟩⟩
def mergeEvent : Nat := 171320
def frameStart : Nat := 171239
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23117⟩⟩] } }
def rhsRaw : List Term := Proof.Events669.exact171287RawTerms
def group : MergeGroup := .relation 171319
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171319) (rhsResult := 171287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23996⟩⟩) ⟨23117⟩ 171287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23117⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23117⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171320

namespace LeftMerge171328
def owner : Owner := ⟨.program ⟨257⟩, ⟨22164⟩⟩
def mergeEvent : Nat := 171328
def frameStart : Nat := 171239
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events669.exact171301RawTerms
def rightRaw : List Term := Proof.Events669.exact171324RawTerms
def group : MergeGroup := .operator 171301 171324
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171301) (leftOrdinal := 0)
    (rightResult := 171324) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22162⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171328

namespace LeftMerge171345
def owner : Owner := ⟨.program ⟨257⟩, ⟨22759⟩⟩
def mergeEvent : Nat := 171345
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }
def rhsRaw : List Term := Proof.Events669.exact171342RawTerms
def group : MergeGroup := .relation 171344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171344) (rhsResult := 171342)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171343 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩) (none) 171342) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171345

namespace LeftMerge171346
def owner : Owner := ⟨.program ⟨257⟩, ⟨22759⟩⟩
def mergeEvent : Nat := 171346
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩] } }
def rhsRaw : List Term := Proof.Events669.exact171342RawTerms
def group : MergeGroup := .relation 171344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171344) (rhsResult := 171342)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171343 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩) (none) 171342) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171346

namespace LeftMerge171347
def owner : Owner := ⟨.program ⟨257⟩, ⟨22759⟩⟩
def mergeEvent : Nat := 171347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23117⟩⟩] } }
def rhsRaw : List Term := Proof.Events669.exact171342RawTerms
def group : MergeGroup := .relation 171344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171344) (rhsResult := 171342)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171343 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩) (none) 171342) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23117⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23117⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171347

namespace LeftMerge171348
def owner : Owner := ⟨.program ⟨257⟩, ⟨22759⟩⟩
def mergeEvent : Nat := 171348
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events669.exact171342RawTerms
def group : MergeGroup := .relation 171344
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171344) (rhsResult := 171342)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171343 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22756⟩⟩]⟩) (none) 171342) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22162⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨22162⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171348

namespace LeftMerge171353
def owner : Owner := ⟨.program ⟨257⟩, ⟨23999⟩⟩
def mergeEvent : Nat := 171353
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩] } }
def leftRaw : List Term := Proof.Events669.exact171349RawTerms
def rightRaw : List Term := Proof.Events668.exact171171RawTerms
def group : MergeGroup := .operator 171349 171171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171349) (leftOrdinal := 0)
    (rightResult := 171171) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23996⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171353

namespace LeftMerge171354
def owner : Owner := ⟨.program ⟨257⟩, ⟨23999⟩⟩
def mergeEvent : Nat := 171354
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23117⟩⟩] } }
def leftRaw : List Term := Proof.Events669.exact171349RawTerms
def rightRaw : List Term := Proof.Events668.exact171171RawTerms
def group : MergeGroup := .operator 171349 171171
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171349) (leftOrdinal := 2)
    (rightResult := 171171) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23117⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23117⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨23117⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171354

namespace LeftMerge171380
def owner : Owner := ⟨.program ⟨257⟩, ⟨18373⟩⟩
def mergeEvent : Nat := 171380
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events031.exact7942RawTerms
def rightRaw : List Term := Proof.Events639.exact163653RawTerms
def group : MergeGroup := .operator 7942 163653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7942) (leftOrdinal := 0)
    (rightResult := 163653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18370⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨18370⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171380

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
