import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge75458
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75458
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 23)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75458

namespace LeftMerge75460
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75460
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15864⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75459
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75459) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15864⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75460

namespace LeftMerge75461
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75461
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 22)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75461

namespace LeftMerge75463
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75463
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15745⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75462
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75462) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15745⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75463

namespace LeftMerge75464
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75464
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 21)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75464

namespace LeftMerge75466
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75466
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15626⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75465
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75465) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15626⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75466

namespace LeftMerge75467
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75467
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 31)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75467

namespace LeftMerge75469
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75469
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17318⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75468) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17318⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75469

namespace LeftMerge75470
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75470
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 20)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75470

namespace LeftMerge75472
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75472
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15362⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75471
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75471) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15362⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75472

namespace LeftMerge75473
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75473
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 19)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75473

namespace LeftMerge75475
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75475
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15306⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75474
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75474) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15306⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75475

namespace LeftMerge75476
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75476
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩] } }
def leftRaw : List Term := Proof.Events294.exact75403RawTerms
def rightRaw : List Term := Proof.Events293.exact75244RawTerms
def group : MergeGroup := .operator 75403 75244
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75403) (leftOrdinal := 18)
    (rightResult := 75244) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18678⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75476

namespace LeftMerge75478
def owner : Owner := ⟨.program ⟨214⟩, ⟨18679⟩⟩
def mergeEvent : Nat := 75478
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15262⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }
def rhsRaw : List Term := Proof.Events293.exact75241RawTerms
def group : MergeGroup := .relation 75477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75477) (rhsResult := 75241)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18678⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18678⟩⟩) ⟨18616⟩ 75241) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15262⟩⟩], [⟨.program ⟨214⟩, ⟨18616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge75478

namespace LeftMerge75486
def owner : Owner := ⟨.program ⟨214⟩, ⟨18493⟩⟩
def mergeEvent : Nat := 75486
def frameStart : Nat := 74728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events293.exact75255RawTerms
def rightRaw : List Term := Proof.Events294.exact75482RawTerms
def group : MergeGroup := .operator 75255 75482
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75255) (leftOrdinal := 0)
    (rightResult := 75482) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18491⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18491⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75486

namespace LeftMerge75503
def owner : Owner := ⟨.program ⟨214⟩, ⟨18558⟩⟩
def mergeEvent : Nat := 75503
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5511⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩] } }
def rhsRaw : List Term := Proof.Events294.exact75500RawTerms
def group : MergeGroup := .relation 75502
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 75502) (rhsResult := 75500)
    (sourceTermOrdinal := 18) (source := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 75501 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨18555⟩⟩]⟩) (none) 75500) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6744⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5511⟩⟩], [⟨.program ⟨214⟩, ⟨6744⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge75503

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
