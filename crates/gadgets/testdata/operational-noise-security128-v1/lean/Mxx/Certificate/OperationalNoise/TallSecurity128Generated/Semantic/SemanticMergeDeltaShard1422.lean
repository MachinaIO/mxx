import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge232265
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232265
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 17)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232265

namespace LeftMerge232266
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232266
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 16)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232266

namespace LeftMerge232267
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232267
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 15)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232267

namespace LeftMerge232268
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232268
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 14)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232268

namespace LeftMerge232269
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232269
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 13)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232269

namespace LeftMerge232270
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232270
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 12)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232270

namespace LeftMerge232271
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232271
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 11)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232271

namespace LeftMerge232272
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232272
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 10)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232272

namespace LeftMerge232273
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232273
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 9)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232273

namespace LeftMerge232274
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232274
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 8)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232274

namespace LeftMerge232275
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232275
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 7)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232275

namespace LeftMerge232276
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232276
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 6)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232276

namespace LeftMerge232277
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232277
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 5)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232277

namespace LeftMerge232278
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232278
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 4)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232278

namespace LeftMerge232279
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232279
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 3)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232279

namespace LeftMerge232280
def owner : Owner := ⟨.program ⟨257⟩, ⟨71205⟩⟩
def mergeEvent : Nat := 232280
def frameStart : Nat := 231586
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩] } }
def leftRaw : List Term := Proof.Events907.exact232261RawTerms
def rightRaw : List Term := Proof.Events906.exact232102RawTerms
def group : MergeGroup := .operator 232261 232102
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 232261) (leftOrdinal := 2)
    (rightResult := 232102) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71204⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge232280

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
