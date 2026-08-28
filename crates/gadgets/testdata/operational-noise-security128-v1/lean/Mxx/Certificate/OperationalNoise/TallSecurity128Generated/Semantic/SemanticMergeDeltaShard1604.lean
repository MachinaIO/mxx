import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge260224
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def mergeEvent : Nat := 260224
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21991⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events981.exact251375RawTerms
def group : MergeGroup := .relation 260223
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 260223) (rhsResult := 251375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21991⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 251375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨21991⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge260224

namespace LeftMerge260225
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def mergeEvent : Nat := 260225
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1016.exact260155RawTerms
def rightRaw : List Term := Proof.Events981.exact251378RawTerms
def group : MergeGroup := .operator 260155 251378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 260155) (leftOrdinal := 1)
    (rightResult := 251378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge260225

namespace LeftMerge260226
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def mergeEvent : Nat := 260226
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1016.exact260155RawTerms
def rightRaw : List Term := Proof.Events981.exact251378RawTerms
def group : MergeGroup := .operator 260155 251378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 260155) (leftOrdinal := 19)
    (rightResult := 251378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge260226

namespace LeftMerge260228
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def mergeEvent : Nat := 260228
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events981.exact251375RawTerms
def group : MergeGroup := .relation 260227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 260227) (rhsResult := 251375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 251375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨18771⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge260228

namespace LeftMerge260229
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def mergeEvent : Nat := 260229
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1016.exact260155RawTerms
def rightRaw : List Term := Proof.Events981.exact251378RawTerms
def group : MergeGroup := .operator 260155 251378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 260155) (leftOrdinal := 0)
    (rightResult := 251378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge260229

namespace LeftMerge260230
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def mergeEvent : Nat := 260230
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1016.exact260155RawTerms
def rightRaw : List Term := Proof.Events981.exact251378RawTerms
def group : MergeGroup := .operator 260155 251378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 260155) (leftOrdinal := 18)
    (rightResult := 251378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge260230

namespace LeftMerge260232
def owner : Owner := ⟨.program ⟨257⟩, ⟨71084⟩⟩
def mergeEvent : Nat := 260232
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events981.exact251375RawTerms
def group : MergeGroup := .relation 260231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 260231) (rhsResult := 251375)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 251375) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge260232

namespace LeftMerge260246
def owner : Owner := ⟨.program ⟨257⟩, ⟨68323⟩⟩
def mergeEvent : Nat := 260246
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1016.exact260240RawTerms
def group : MergeGroup := .operator 251495 260240
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 260240) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68320⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68320⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge260246

namespace LeftMerge261367
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261367
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48298⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261367

namespace LeftMerge261368
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261368
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 10) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261368

namespace LeftMerge261369
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261369
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42934⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42934⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42934⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261369

namespace LeftMerge261370
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261370
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261370

namespace LeftMerge261371
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261371
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37578⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37578⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261371

namespace LeftMerge261372
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261372
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 6) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34898⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261372

namespace LeftMerge261373
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261373
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 4) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29234⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261373

namespace LeftMerge261374
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def mergeEvent : Nat := 261374
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1020.exact261363RawTerms
def rightRaw : List Term := Proof.Events1020.exact261361RawTerms
def group : MergeGroup := .operator 261363 261361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261363) (leftOrdinal := 0)
    (rightResult := 261361) (rightOrdinal := 3) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26554⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261374

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
