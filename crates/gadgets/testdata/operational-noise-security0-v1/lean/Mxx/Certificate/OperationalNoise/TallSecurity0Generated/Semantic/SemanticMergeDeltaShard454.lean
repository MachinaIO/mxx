import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge75415
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75415
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 9)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6727⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75415

namespace LeftMerge75416
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75416
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 8)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6725⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6725⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75416

namespace LeftMerge75417
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75417
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 7)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6723⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6723⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75417

namespace LeftMerge75418
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75418
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 6)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6721⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6721⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75418

namespace LeftMerge75419
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75419
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 5)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75419

namespace LeftMerge75420
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75420
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 4)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75420

namespace LeftMerge75421
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75421
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 3)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75421

namespace LeftMerge75422
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75422
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 2)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75422

namespace LeftMerge75423
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75423
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 1)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75423

namespace LeftMerge75424
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75424
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 0)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6709⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6709⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75424

namespace LeftMerge75425
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75425
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 33)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18167⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75425

namespace LeftMerge75427
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75427
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18167⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75426
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75426) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18167⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18167⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75427

namespace LeftMerge75428
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75428
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 29)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17082⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75428

namespace LeftMerge75430
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75430
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17082⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75429) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17082⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17082⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75430

namespace LeftMerge75431
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75431
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 28)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75431

namespace LeftMerge75433
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75433
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16795⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75432) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16795⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75433

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
