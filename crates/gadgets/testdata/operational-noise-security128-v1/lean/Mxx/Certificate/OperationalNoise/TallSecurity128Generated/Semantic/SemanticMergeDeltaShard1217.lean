import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge198480
def owner : Owner := ⟨.program ⟨257⟩, ⟨58503⟩⟩
def mergeEvent : Nat := 198480
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57981⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198476RawTerms
def rightRaw : List Term := Proof.Events774.exact198290RawTerms
def group : MergeGroup := .operator 198476 198290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198476) (leftOrdinal := 2)
    (rightResult := 198290) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57981⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨57981⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25034⟩⟩, ⟨.program ⟨257⟩, ⟨56559⟩⟩], [⟨.program ⟨257⟩, ⟨57981⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198480

namespace LeftMerge198481
def owner : Owner := ⟨.program ⟨257⟩, ⟨58503⟩⟩
def mergeEvent : Nat := 198481
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198476RawTerms
def rightRaw : List Term := Proof.Events774.exact198290RawTerms
def group : MergeGroup := .operator 198476 198290
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198476) (leftOrdinal := 1)
    (rightResult := 198290) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7290⟩⟩, ⟨.program ⟨257⟩, ⟨9532⟩⟩, ⟨.program ⟨257⟩, ⟨58501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198481

namespace LeftMerge198489
def owner : Owner := ⟨.program ⟨257⟩, ⟨58976⟩⟩
def mergeEvent : Nat := 198489
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198483RawTerms
def rightRaw : List Term := Proof.Events774.exact198206RawTerms
def group : MergeGroup := .operator 198483 198206
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198483) (leftOrdinal := 0)
    (rightResult := 198206) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58974⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198489

namespace LeftMerge198490
def owner : Owner := ⟨.program ⟨257⟩, ⟨58976⟩⟩
def mergeEvent : Nat := 198490
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198483RawTerms
def rightRaw : List Term := Proof.Events774.exact198206RawTerms
def group : MergeGroup := .operator 198483 198206
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198483) (leftOrdinal := 1)
    (rightResult := 198206) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58974⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198490

namespace LeftMerge198492
def owner : Owner := ⟨.program ⟨257⟩, ⟨58976⟩⟩
def mergeEvent : Nat := 198492
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58139⟩⟩] } }
def rhsRaw : List Term := Proof.Events774.exact198203RawTerms
def group : MergeGroup := .relation 198491
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198491) (rhsResult := 198203)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58974⟩⟩) ⟨58139⟩ 198203) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58139⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨58139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198492

namespace LeftMerge198506
def owner : Owner := ⟨.program ⟨257⟩, ⟨57759⟩⟩
def mergeEvent : Nat := 198506
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events775.exact198500RawTerms
def group : MergeGroup := .operator 192995 198500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 198500) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨57756⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198506

namespace LeftMerge198627
def owner : Owner := ⟨.program ⟨257⟩, ⟨58336⟩⟩
def mergeEvent : Nat := 198627
def frameStart : Nat := 198561
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198623RawTerms
def rightRaw : List Term := Proof.Events775.exact198621RawTerms
def group : MergeGroup := .operator 198623 198621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198623) (leftOrdinal := 0)
    (rightResult := 198621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198627

namespace LeftMerge198639
def owner : Owner := ⟨.program ⟨257⟩, ⟨58975⟩⟩
def mergeEvent : Nat := 198639
def frameStart : Nat := 198561
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198635RawTerms
def rightRaw : List Term := Proof.Events775.exact198612RawTerms
def group : MergeGroup := .operator 198635 198612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198635) (leftOrdinal := 0)
    (rightResult := 198612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58974⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198639

namespace LeftMerge198640
def owner : Owner := ⟨.program ⟨257⟩, ⟨58975⟩⟩
def mergeEvent : Nat := 198640
def frameStart : Nat := 198561
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198635RawTerms
def rightRaw : List Term := Proof.Events775.exact198612RawTerms
def group : MergeGroup := .operator 198635 198612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198635) (leftOrdinal := 1)
    (rightResult := 198612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58974⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198640

namespace LeftMerge198642
def owner : Owner := ⟨.program ⟨257⟩, ⟨58975⟩⟩
def mergeEvent : Nat := 198642
def frameStart : Nat := 198561
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58139⟩⟩] } }
def rhsRaw : List Term := Proof.Events775.exact198609RawTerms
def group : MergeGroup := .relation 198641
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198641) (rhsResult := 198609)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨58974⟩⟩) ⟨58139⟩ 198609) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨58139⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨58139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198642

namespace LeftMerge198650
def owner : Owner := ⟨.program ⟨257⟩, ⟨57161⟩⟩
def mergeEvent : Nat := 198650
def frameStart : Nat := 198561
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events775.exact198623RawTerms
def rightRaw : List Term := Proof.Events775.exact198646RawTerms
def group : MergeGroup := .operator 198623 198646
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198623) (leftOrdinal := 0)
    (rightResult := 198646) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198650

namespace LeftMerge198667
def owner : Owner := ⟨.program ⟨257⟩, ⟨57759⟩⟩
def mergeEvent : Nat := 198667
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }
def rhsRaw : List Term := Proof.Events776.exact198664RawTerms
def group : MergeGroup := .relation 198666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198666) (rhsResult := 198664)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198665 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩]⟩) (none) 198664) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7210⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198667

namespace LeftMerge198668
def owner : Owner := ⟨.program ⟨257⟩, ⟨57759⟩⟩
def mergeEvent : Nat := 198668
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩] } }
def rhsRaw : List Term := Proof.Events776.exact198664RawTerms
def group : MergeGroup := .relation 198666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198666) (rhsResult := 198664)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198665 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩]⟩) (none) 198664) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198668

namespace LeftMerge198669
def owner : Owner := ⟨.program ⟨257⟩, ⟨57759⟩⟩
def mergeEvent : Nat := 198669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58139⟩⟩] } }
def rhsRaw : List Term := Proof.Events776.exact198664RawTerms
def group : MergeGroup := .relation 198666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198666) (rhsResult := 198664)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198665 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩]⟩) (none) 198664) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨56864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58139⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨56864⟩⟩], [⟨.program ⟨257⟩, ⟨58139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198669

namespace LeftMerge198670
def owner : Owner := ⟨.program ⟨257⟩, ⟨57759⟩⟩
def mergeEvent : Nat := 198670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events776.exact198664RawTerms
def group : MergeGroup := .relation 198666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 198666) (rhsResult := 198664)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 198665 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨57756⟩⟩]⟩) (none) 198664) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨57159⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨57159⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge198670

namespace LeftMerge198675
def owner : Owner := ⟨.program ⟨257⟩, ⟨58977⟩⟩
def mergeEvent : Nat := 198675
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩] } }
def leftRaw : List Term := Proof.Events776.exact198671RawTerms
def rightRaw : List Term := Proof.Events775.exact198493RawTerms
def group : MergeGroup := .operator 198671 198493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 198671) (leftOrdinal := 0)
    (rightResult := 198493) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58974⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge198675

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
