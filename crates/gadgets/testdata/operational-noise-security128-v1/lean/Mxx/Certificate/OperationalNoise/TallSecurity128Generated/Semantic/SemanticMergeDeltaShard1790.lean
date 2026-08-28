import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge290741
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290741
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 5)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290741

namespace LeftMerge290742
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290742
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 4)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290742

namespace LeftMerge290743
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290743
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 3)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290743

namespace LeftMerge290744
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290744
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 2)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290744

namespace LeftMerge290745
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290745
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 1)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290745

namespace LeftMerge290746
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290746
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 0)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge290746

namespace LeftMerge290747
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290747
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48285⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 29)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48285⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290747

namespace LeftMerge290749
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290749
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48285⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1135.exact290563RawTerms
def group : MergeGroup := .relation 290748
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 290748) (rhsResult := 290563)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48285⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290749

namespace LeftMerge290750
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290750
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45605⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 28)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45605⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290750

namespace LeftMerge290752
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290752
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45605⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1135.exact290563RawTerms
def group : MergeGroup := .relation 290751
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 290751) (rhsResult := 290563)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45605⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290752

namespace LeftMerge290753
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290753
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42921⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 27)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42921⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290753

namespace LeftMerge290755
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290755
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42921⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1135.exact290563RawTerms
def group : MergeGroup := .relation 290754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 290754) (rhsResult := 290563)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42921⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290755

namespace LeftMerge290756
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290756
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40241⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 26)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40241⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290756

namespace LeftMerge290758
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290758
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40241⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1135.exact290563RawTerms
def group : MergeGroup := .relation 290757
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 290757) (rhsResult := 290563)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40241⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290758

namespace LeftMerge290759
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290759
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩] } }
def leftRaw : List Term := Proof.Events1135.exact290725RawTerms
def rightRaw : List Term := Proof.Events1135.exact290566RawTerms
def group : MergeGroup := .operator 290725 290566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 290725) (leftOrdinal := 25)
    (rightResult := 290566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71048⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290759

namespace LeftMerge290761
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def mergeEvent : Nat := 290761
def frameStart : Nat := 290050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37565⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }
def rhsRaw : List Term := Proof.Events1135.exact290563RawTerms
def group : MergeGroup := .relation 290760
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 290760) (rhsResult := 290563)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71048⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71048⟩⟩) ⟨68794⟩ 290563) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68794⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37565⟩⟩], [⟨.program ⟨257⟩, ⟨68794⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge290761

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
