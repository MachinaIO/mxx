import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge304335
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304335
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 9)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304335

namespace LeftMerge304336
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304336
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 8)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304336

namespace LeftMerge304337
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304337
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 7)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304337

namespace LeftMerge304338
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304338
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 6)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304338

namespace LeftMerge304339
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304339
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 5)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304339

namespace LeftMerge304340
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304340
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 4)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304340

namespace LeftMerge304341
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304341
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 3)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304341

namespace LeftMerge304342
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304342
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 2)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304342

namespace LeftMerge304343
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304343
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 1)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304343

namespace LeftMerge304344
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304344
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 0)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304344

namespace LeftMerge304345
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304345
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 29)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304345

namespace LeftMerge304347
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304347
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304346) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304347

namespace LeftMerge304348
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304348
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 28)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304348

namespace LeftMerge304350
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304350
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304349) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304350

namespace LeftMerge304351
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304351
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42869⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1188.exact304323RawTerms
def rightRaw : List Term := Proof.Events1188.exact304164RawTerms
def group : MergeGroup := .operator 304323 304164
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304323) (leftOrdinal := 27)
    (rightResult := 304164) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42869⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304351

namespace LeftMerge304353
def owner : Owner := ⟨.program ⟨257⟩, ⟨70935⟩⟩
def mergeEvent : Nat := 304353
def frameStart : Nat := 303660
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42869⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1188.exact304161RawTerms
def group : MergeGroup := .relation 304352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304352) (rhsResult := 304161)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 304161) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304353

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
