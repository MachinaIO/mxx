import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge31401
def owner : Owner := ⟨.program ⟨214⟩, ⟨18661⟩⟩
def mergeEvent : Nat := 31401
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events122.exact31380RawTerms
def rightRaw : List Term := Proof.Events122.exact31378RawTerms
def group : MergeGroup := .operator 31380 31378
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31380) (leftOrdinal := 0)
    (rightResult := 31378) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15274⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15274⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31401

namespace LeftMerge31532
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31532
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 17)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6743⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6743⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31532

namespace LeftMerge31533
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31533
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 16)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6741⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6741⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31533

namespace LeftMerge31534
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31534
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 15)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6739⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6739⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31534

namespace LeftMerge31535
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31535
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 14)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6737⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6737⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31535

namespace LeftMerge31536
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31536
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 13)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6735⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6735⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31536

namespace LeftMerge31537
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31537
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 12)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6733⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6733⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31537

namespace LeftMerge31538
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31538
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 11)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6731⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6731⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31538

namespace LeftMerge31539
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31539
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 10)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6729⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6729⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31539

namespace LeftMerge31540
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31540
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 9)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31540

namespace LeftMerge31541
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31541
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 8)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31541

namespace LeftMerge31542
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31542
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 7)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31542

namespace LeftMerge31543
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31543
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 6)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31543

namespace LeftMerge31544
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31544
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 5)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31544

namespace LeftMerge31545
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31545
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 4)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31545

namespace LeftMerge31546
def owner : Owner := ⟨.program ⟨214⟩, ⟨18691⟩⟩
def mergeEvent : Nat := 31546
def frameStart : Nat := 30853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩] } }
def leftRaw : List Term := Proof.Events123.exact31528RawTerms
def rightRaw : List Term := Proof.Events122.exact31369RawTerms
def group : MergeGroup := .operator 31528 31369
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 31528) (leftOrdinal := 3)
    (rightResult := 31369) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18690⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18690⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge31546

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
