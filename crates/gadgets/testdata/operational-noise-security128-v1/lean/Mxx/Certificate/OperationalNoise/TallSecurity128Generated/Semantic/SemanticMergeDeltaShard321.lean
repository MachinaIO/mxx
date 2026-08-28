import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge56777
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56777
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 5)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56777

namespace LeftMerge56778
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56778
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 4)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56778

namespace LeftMerge56779
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56779
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 3)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56779

namespace LeftMerge56780
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56780
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 2)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56780

namespace LeftMerge56781
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56781
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 1)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56781

namespace LeftMerge56782
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56782
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 0)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56782

namespace LeftMerge56783
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56783
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48467⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 29)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48467⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56783

namespace LeftMerge56785
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56785
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48467⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56784) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48467⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56785

namespace LeftMerge56786
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56786
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45787⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 28)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45787⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45787⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56786

namespace LeftMerge56788
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56788
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45787⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56787) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45787⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45787⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56788

namespace LeftMerge56789
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56789
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 27)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨43103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56789

namespace LeftMerge56791
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56791
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43103⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56790) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨43103⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56791

namespace LeftMerge56792
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56792
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40423⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 26)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40423⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56792

namespace LeftMerge56794
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56794
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40423⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56793) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40423⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56794

namespace LeftMerge56795
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56795
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37747⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 25)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37747⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56795

namespace LeftMerge56797
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56797
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37747⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56796
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56796) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37747⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56797

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
