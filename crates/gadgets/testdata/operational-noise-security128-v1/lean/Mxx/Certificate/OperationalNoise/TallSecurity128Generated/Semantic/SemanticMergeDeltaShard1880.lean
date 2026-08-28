import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge304354
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304354
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40189⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 26)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40189⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304354

namespace LeftMerge304356
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304356
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40189⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304355
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304355) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304356

namespace LeftMerge304357
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304357
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37513⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 25)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37513⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304357

namespace LeftMerge304359
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304359
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37513⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304358
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304358) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304359

namespace LeftMerge304360
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304360
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34833⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 24)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34833⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304360

namespace LeftMerge304362
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304362
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34833⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304361
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304361) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304362

namespace LeftMerge304363
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304363
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29169⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 22)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29169⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304363

namespace LeftMerge304365
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304365
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29169⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304364
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304364) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29169⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304365

namespace LeftMerge304366
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304366
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26489⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 21)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26489⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304366

namespace LeftMerge304368
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304368
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26489⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304367
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304367) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26489⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304368

namespace LeftMerge304369
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304369
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65901⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 35)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65901⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304369

namespace LeftMerge304371
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304371
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65901⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304370
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304370) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65901⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304371

namespace LeftMerge304372
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304372
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62891⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 34)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62891⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304372

namespace LeftMerge304374
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304374
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62891⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304373
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304373) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62891⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304374

namespace LeftMerge304375
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304375
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 33)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304375

namespace LeftMerge304377
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304377
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59911⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304376
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304376) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59911⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304377

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
