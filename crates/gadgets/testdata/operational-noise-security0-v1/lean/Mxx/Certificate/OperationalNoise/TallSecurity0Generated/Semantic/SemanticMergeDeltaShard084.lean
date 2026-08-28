import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge15600
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6416RawTerms
def group : MergeGroup := .relation 15599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 15599) (rhsResult := 6416)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15760⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15600

namespace LeftMerge15601
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15601
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 5)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6719⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6719⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15601

namespace LeftMerge15602
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15602
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 21)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15602

namespace LeftMerge15604
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15604
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6416RawTerms
def group : MergeGroup := .relation 15603
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 15603) (rhsResult := 6416)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15641⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15604

namespace LeftMerge15605
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15605
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 4)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6717⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6717⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15605

namespace LeftMerge15606
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15606
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 31)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15606

namespace LeftMerge15608
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15608
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6416RawTerms
def group : MergeGroup := .relation 15607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 15607) (rhsResult := 6416)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨17363⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15608

namespace LeftMerge15609
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15609
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 3)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6715⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6715⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15609

namespace LeftMerge15610
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15610
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 20)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15610

namespace LeftMerge15612
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15612
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6416RawTerms
def group : MergeGroup := .relation 15611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 15611) (rhsResult := 6416)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15382⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15612

namespace LeftMerge15613
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15613
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 2)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6713⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6713⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15613

namespace LeftMerge15614
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15614
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 19)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15614

namespace LeftMerge15616
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15616
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6416RawTerms
def group : MergeGroup := .relation 15615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 15615) (rhsResult := 6416)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15616

namespace LeftMerge15617
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15617
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 1)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge15617

namespace LeftMerge15618
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15618
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩] } }
def leftRaw : List Term := Proof.Events060.exact15544RawTerms
def rightRaw : List Term := Proof.Events025.exact6419RawTerms
def group : MergeGroup := .operator 15544 6419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 15544) (leftOrdinal := 18)
    (rightResult := 6419) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18693⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15618

namespace LeftMerge15620
def owner : Owner := ⟨.program ⟨214⟩, ⟨30210⟩⟩
def mergeEvent : Nat := 15620
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }
def rhsRaw : List Term := Proof.Events025.exact6416RawTerms
def group : MergeGroup := .relation 15619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 15619) (rhsResult := 6416)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18693⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18693⟩⟩) ⟨18626⟩ 6416) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18626⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15277⟩⟩], [⟨.program ⟨214⟩, ⟨18626⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge15620

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
