import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge165654
def owner : Owner := ⟨.program ⟨257⟩, ⟨37217⟩⟩
def mergeEvent : Nat := 165654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def leftRaw : List Term := Proof.Events647.exact165650RawTerms
def rightRaw : List Term := Proof.Events646.exact165620RawTerms
def group : MergeGroup := .operator 165650 165620
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165650) (leftOrdinal := 1)
    (rightResult := 165620) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165654

namespace LeftMerge165662
def owner : Owner := ⟨.program ⟨257⟩, ⟨38984⟩⟩
def mergeEvent : Nat := 165662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩] } }
def leftRaw : List Term := Proof.Events647.exact165656RawTerms
def rightRaw : List Term := Proof.Events646.exact165592RawTerms
def group : MergeGroup := .operator 165656 165592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165656) (leftOrdinal := 1)
    (rightResult := 165592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38983⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165662

namespace LeftMerge165664
def owner : Owner := ⟨.program ⟨257⟩, ⟨38984⟩⟩
def mergeEvent : Nat := 165664
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38453⟩⟩] } }
def rhsRaw : List Term := Proof.Events646.exact165589RawTerms
def group : MergeGroup := .relation 165663
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165663) (rhsResult := 165589)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38983⟩⟩) ⟨38453⟩ 165589) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38453⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], [⟨.program ⟨257⟩, ⟨38453⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165664

namespace LeftMerge165665
def owner : Owner := ⟨.program ⟨257⟩, ⟨38984⟩⟩
def mergeEvent : Nat := 165665
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩] } }
def leftRaw : List Term := Proof.Events647.exact165656RawTerms
def rightRaw : List Term := Proof.Events646.exact165592RawTerms
def group : MergeGroup := .operator 165656 165592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165656) (leftOrdinal := 0)
    (rightResult := 165592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38983⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165665

namespace LeftMerge165679
def owner : Owner := ⟨.program ⟨257⟩, ⟨37912⟩⟩
def mergeEvent : Nat := 165679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events647.exact165673RawTerms
def group : MergeGroup := .operator 163745 165673
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 165673) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37909⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165679

namespace LeftMerge165758
def owner : Owner := ⟨.program ⟨257⟩, ⟨37211⟩⟩
def mergeEvent : Nat := 165758
def frameStart : Nat := 165728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events647.exact165754RawTerms
def rightRaw : List Term := Proof.Events647.exact165751RawTerms
def group : MergeGroup := .operator 165754 165751
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165754) (leftOrdinal := 0)
    (rightResult := 165751) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13941⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165758

namespace LeftMerge165788
def owner : Owner := ⟨.program ⟨257⟩, ⟨38724⟩⟩
def mergeEvent : Nat := 165788
def frameStart : Nat := 165728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events647.exact165784RawTerms
def rightRaw : List Term := Proof.Events647.exact165782RawTerms
def group : MergeGroup := .operator 165784 165782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165784) (leftOrdinal := 0)
    (rightResult := 165782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165788

namespace LeftMerge165811
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 165811
def frameStart : Nat := 165728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events647.exact165807RawTerms
def rightRaw : List Term := Proof.Events647.exact165804RawTerms
def group : MergeGroup := .operator 165807 165804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165807) (leftOrdinal := 0)
    (rightResult := 165804) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165811

namespace LeftMerge165820
def owner : Owner := ⟨.program ⟨257⟩, ⟨38986⟩⟩
def mergeEvent : Nat := 165820
def frameStart : Nat := 165728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩] } }
def leftRaw : List Term := Proof.Events647.exact165816RawTerms
def rightRaw : List Term := Proof.Events647.exact165773RawTerms
def group : MergeGroup := .operator 165816 165773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165816) (leftOrdinal := 0)
    (rightResult := 165773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38983⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165820

namespace LeftMerge165821
def owner : Owner := ⟨.program ⟨257⟩, ⟨38986⟩⟩
def mergeEvent : Nat := 165821
def frameStart : Nat := 165728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩] } }
def leftRaw : List Term := Proof.Events647.exact165816RawTerms
def rightRaw : List Term := Proof.Events647.exact165773RawTerms
def group : MergeGroup := .operator 165816 165773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165816) (leftOrdinal := 1)
    (rightResult := 165773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38983⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165821

namespace LeftMerge165823
def owner : Owner := ⟨.program ⟨257⟩, ⟨38986⟩⟩
def mergeEvent : Nat := 165823
def frameStart : Nat := 165728
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38453⟩⟩] } }
def rhsRaw : List Term := Proof.Events647.exact165770RawTerms
def group : MergeGroup := .relation 165822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165822) (rhsResult := 165770)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38983⟩⟩) ⟨38453⟩ 165770) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38453⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], [⟨.program ⟨257⟩, ⟨38453⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165823

namespace LeftMerge165831
def owner : Owner := ⟨.program ⟨257⟩, ⟨37462⟩⟩
def mergeEvent : Nat := 165831
def frameStart : Nat := 165728
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events647.exact165784RawTerms
def rightRaw : List Term := Proof.Events647.exact165827RawTerms
def group : MergeGroup := .operator 165784 165827
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 165784) (leftOrdinal := 0)
    (rightResult := 165827) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37460⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37460⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165831

namespace LeftMerge165848
def owner : Owner := ⟨.program ⟨257⟩, ⟨37912⟩⟩
def mergeEvent : Nat := 165848
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events647.exact165845RawTerms
def group : MergeGroup := .relation 165847
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165847) (rhsResult := 165845)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 165846 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩]⟩) (none) 165845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165848

namespace LeftMerge165849
def owner : Owner := ⟨.program ⟨257⟩, ⟨37912⟩⟩
def mergeEvent : Nat := 165849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩] } }
def rhsRaw : List Term := Proof.Events647.exact165845RawTerms
def group : MergeGroup := .relation 165847
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165847) (rhsResult := 165845)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 165846 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩]⟩) (none) 165845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165849

namespace LeftMerge165850
def owner : Owner := ⟨.program ⟨257⟩, ⟨37912⟩⟩
def mergeEvent : Nat := 165850
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38453⟩⟩] } }
def rhsRaw : List Term := Proof.Events647.exact165845RawTerms
def group : MergeGroup := .relation 165847
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165847) (rhsResult := 165845)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 165846 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩]⟩) (none) 165845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38453⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨13941⟩⟩, ⟨.program ⟨257⟩, ⟨37210⟩⟩], [⟨.program ⟨257⟩, ⟨38453⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge165850

namespace LeftMerge165851
def owner : Owner := ⟨.program ⟨257⟩, ⟨37912⟩⟩
def mergeEvent : Nat := 165851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events647.exact165845RawTerms
def group : MergeGroup := .relation 165847
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 165847) (rhsResult := 165845)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 165846 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37909⟩⟩]⟩) (none) 165845) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37460⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨37460⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge165851

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
