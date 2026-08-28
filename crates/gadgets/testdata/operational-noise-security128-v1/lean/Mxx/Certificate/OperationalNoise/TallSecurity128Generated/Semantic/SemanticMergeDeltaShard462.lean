import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge77784
def owner : Owner := ⟨.program ⟨257⟩, ⟨42140⟩⟩
def mergeEvent : Nat := 77784
def frameStart : Nat := 77705
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩] } }
def leftRaw : List Term := Proof.Events303.exact77779RawTerms
def rightRaw : List Term := Proof.Events303.exact77756RawTerms
def group : MergeGroup := .operator 77779 77756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77779) (leftOrdinal := 1)
    (rightResult := 77756) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨42139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77784

namespace LeftMerge77786
def owner : Owner := ⟨.program ⟨257⟩, ⟨42140⟩⟩
def mergeEvent : Nat := 77786
def frameStart : Nat := 77705
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41315⟩⟩] } }
def rhsRaw : List Term := Proof.Events303.exact77753RawTerms
def group : MergeGroup := .relation 77785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77785) (rhsResult := 77753)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨42139⟩⟩) ⟨41315⟩ 77753) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨41315⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77786

namespace LeftMerge77794
def owner : Owner := ⟨.program ⟨257⟩, ⟨40398⟩⟩
def mergeEvent : Nat := 77794
def frameStart : Nat := 77705
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events303.exact77767RawTerms
def rightRaw : List Term := Proof.Events303.exact77790RawTerms
def group : MergeGroup := .operator 77767 77790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77767) (leftOrdinal := 0)
    (rightResult := 77790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77794

namespace LeftMerge77811
def owner : Owner := ⟨.program ⟨257⟩, ⟨40979⟩⟩
def mergeEvent : Nat := 77811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }
def rhsRaw : List Term := Proof.Events303.exact77808RawTerms
def group : MergeGroup := .relation 77810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77810) (rhsResult := 77808)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩) (none) 77808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77811

namespace LeftMerge77812
def owner : Owner := ⟨.program ⟨257⟩, ⟨40979⟩⟩
def mergeEvent : Nat := 77812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩] } }
def rhsRaw : List Term := Proof.Events303.exact77808RawTerms
def group : MergeGroup := .relation 77810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77810) (rhsResult := 77808)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩) (none) 77808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77812

namespace LeftMerge77813
def owner : Owner := ⟨.program ⟨257⟩, ⟨40979⟩⟩
def mergeEvent : Nat := 77813
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41315⟩⟩] } }
def rhsRaw : List Term := Proof.Events303.exact77808RawTerms
def group : MergeGroup := .relation 77810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77810) (rhsResult := 77808)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩) (none) 77808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41315⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77813

namespace LeftMerge77814
def owner : Owner := ⟨.program ⟨257⟩, ⟨40979⟩⟩
def mergeEvent : Nat := 77814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events303.exact77808RawTerms
def group : MergeGroup := .relation 77810
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 77810) (rhsResult := 77808)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 77809 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨40976⟩⟩]⟩) (none) 77808) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40397⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40397⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77814

namespace LeftMerge77819
def owner : Owner := ⟨.program ⟨257⟩, ⟨42142⟩⟩
def mergeEvent : Nat := 77819
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩] } }
def leftRaw : List Term := Proof.Events303.exact77815RawTerms
def rightRaw : List Term := Proof.Events303.exact77637RawTerms
def group : MergeGroup := .operator 77815 77637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77815) (leftOrdinal := 0)
    (rightResult := 77637) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7193⟩⟩, ⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77819

namespace LeftMerge77820
def owner : Owner := ⟨.program ⟨257⟩, ⟨42142⟩⟩
def mergeEvent : Nat := 77820
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41315⟩⟩] } }
def leftRaw : List Term := Proof.Events303.exact77815RawTerms
def rightRaw : List Term := Proof.Events303.exact77637RawTerms
def group : MergeGroup := .operator 77815 77637
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77815) (leftOrdinal := 2)
    (rightResult := 77637) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41315⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨41315⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40156⟩⟩], [⟨.program ⟨257⟩, ⟨41315⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77820

namespace LeftMerge77846
def owner : Owner := ⟨.program ⟨257⟩, ⟨37261⟩⟩
def mergeEvent : Nat := 77846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events012.exact3178RawTerms
def rightRaw : List Term := Proof.Events296.exact75903RawTerms
def group : MergeGroup := .operator 3178 75903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3178) (leftOrdinal := 0)
    (rightResult := 75903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77846

namespace LeftMerge77851
def owner : Owner := ⟨.program ⟨257⟩, ⟨10339⟩⟩
def mergeEvent : Nat := 77851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75773RawTerms
def rightRaw : List Term := Proof.Events074.exact19084RawTerms
def group : MergeGroup := .operator 75773 19084
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75773) (leftOrdinal := 0)
    (rightResult := 19084) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77851

namespace LeftMerge77868
def owner : Owner := ⟨.program ⟨257⟩, ⟨37264⟩⟩
def mergeEvent : Nat := 77868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact77862RawTerms
def rightRaw : List Term := Proof.Events012.exact3181RawTerms
def group : MergeGroup := .operator 77862 3181
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77862) (leftOrdinal := 1)
    (rightResult := 3181) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩, ⟨.program ⟨257⟩, ⟨37258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77868

namespace LeftMerge77869
def owner : Owner := ⟨.program ⟨257⟩, ⟨37264⟩⟩
def mergeEvent : Nat := 77869
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact77862RawTerms
def rightRaw : List Term := Proof.Events012.exact3181RawTerms
def group : MergeGroup := .operator 77862 3181
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77862) (leftOrdinal := 0)
    (rightResult := 3181) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77869

namespace LeftMerge77874
def owner : Owner := ⟨.program ⟨257⟩, ⟨13972⟩⟩
def mergeEvent : Nat := 77874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events012.exact3181RawTerms
def rightRaw : List Term := Proof.Events296.exact75903RawTerms
def group : MergeGroup := .operator 3181 75903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 3181) (leftOrdinal := 0)
    (rightResult := 75903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77874

namespace LeftMerge77879
def owner : Owner := ⟨.program ⟨257⟩, ⟨10356⟩⟩
def mergeEvent : Nat := 77879
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }
def leftRaw : List Term := Proof.Events295.exact75773RawTerms
def rightRaw : List Term := Proof.Events074.exact19125RawTerms
def group : MergeGroup := .operator 75773 19125
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75773) (leftOrdinal := 0)
    (rightResult := 19125) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge77879

namespace LeftMerge77896
def owner : Owner := ⟨.program ⟨257⟩, ⟨13975⟩⟩
def mergeEvent : Nat := 77896
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events304.exact77890RawTerms
def rightRaw : List Term := Proof.Events074.exact19114RawTerms
def group : MergeGroup := .operator 77890 19114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 77890) (leftOrdinal := 1)
    (rightResult := 19114) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨13971⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge77896

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
