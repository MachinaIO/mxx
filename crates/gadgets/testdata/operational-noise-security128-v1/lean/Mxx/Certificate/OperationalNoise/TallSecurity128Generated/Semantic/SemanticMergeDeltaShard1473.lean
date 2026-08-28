import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge238774
def owner : Owner := ⟨.program ⟨257⟩, ⟨13855⟩⟩
def mergeEvent : Nat := 238774
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events932.exact238765RawTerms
def rightRaw : List Term := Proof.Events074.exact19114RawTerms
def group : MergeGroup := .operator 238765 19114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238765) (leftOrdinal := 0)
    (rightResult := 19114) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238774

namespace LeftMerge238779
def owner : Owner := ⟨.program ⟨257⟩, ⟨37073⟩⟩
def mergeEvent : Nat := 238779
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }
def leftRaw : List Term := Proof.Events932.exact238775RawTerms
def rightRaw : List Term := Proof.Events932.exact238745RawTerms
def group : MergeGroup := .operator 238775 238745
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238775) (leftOrdinal := 1)
    (rightResult := 238745) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7281⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩], [⟨.program ⟨257⟩, ⟨7281⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238779

namespace LeftMerge238787
def owner : Owner := ⟨.program ⟨257⟩, ⟨38918⟩⟩
def mergeEvent : Nat := 238787
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩] } }
def leftRaw : List Term := Proof.Events932.exact238781RawTerms
def rightRaw : List Term := Proof.Events932.exact238717RawTerms
def group : MergeGroup := .operator 238781 238717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238781) (leftOrdinal := 1)
    (rightResult := 238717) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38917⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238787

namespace LeftMerge238789
def owner : Owner := ⟨.program ⟨257⟩, ⟨38918⟩⟩
def mergeEvent : Nat := 238789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38417⟩⟩] } }
def rhsRaw : List Term := Proof.Events932.exact238714RawTerms
def group : MergeGroup := .relation 238788
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238788) (rhsResult := 238714)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38917⟩⟩) ⟨38417⟩ 238714) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38417⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], [⟨.program ⟨257⟩, ⟨38417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238789

namespace LeftMerge238790
def owner : Owner := ⟨.program ⟨257⟩, ⟨38918⟩⟩
def mergeEvent : Nat := 238790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩] } }
def leftRaw : List Term := Proof.Events932.exact238781RawTerms
def rightRaw : List Term := Proof.Events932.exact238717RawTerms
def group : MergeGroup := .operator 238781 238717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238781) (leftOrdinal := 0)
    (rightResult := 238717) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38917⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238790

namespace LeftMerge238804
def owner : Owner := ⟨.program ⟨257⟩, ⟨37852⟩⟩
def mergeEvent : Nat := 238804
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37849⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events932.exact238798RawTerms
def group : MergeGroup := .operator 236870 238798
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 238798) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨37849⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37849⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238804

namespace LeftMerge238883
def owner : Owner := ⟨.program ⟨257⟩, ⟨37067⟩⟩
def mergeEvent : Nat := 238883
def frameStart : Nat := 238853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events933.exact238879RawTerms
def rightRaw : List Term := Proof.Events933.exact238876RawTerms
def group : MergeGroup := .operator 238879 238876
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238879) (leftOrdinal := 0)
    (rightResult := 238876) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238883

namespace LeftMerge238913
def owner : Owner := ⟨.program ⟨257⟩, ⟨38700⟩⟩
def mergeEvent : Nat := 238913
def frameStart : Nat := 238853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events933.exact238909RawTerms
def rightRaw : List Term := Proof.Events933.exact238907RawTerms
def group : MergeGroup := .operator 238909 238907
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238909) (leftOrdinal := 0)
    (rightResult := 238907) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238913

namespace LeftMerge238936
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def mergeEvent : Nat := 238936
def frameStart : Nat := 238853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }
def leftRaw : List Term := Proof.Events933.exact238932RawTerms
def rightRaw : List Term := Proof.Events933.exact238929RawTerms
def group : MergeGroup := .operator 238932 238929
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238932) (leftOrdinal := 0)
    (rightResult := 238929) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9553⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238936

namespace LeftMerge238945
def owner : Owner := ⟨.program ⟨257⟩, ⟨38920⟩⟩
def mergeEvent : Nat := 238945
def frameStart : Nat := 238853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩] } }
def leftRaw : List Term := Proof.Events933.exact238941RawTerms
def rightRaw : List Term := Proof.Events933.exact238898RawTerms
def group : MergeGroup := .operator 238941 238898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238941) (leftOrdinal := 0)
    (rightResult := 238898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38917⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238945

namespace LeftMerge238946
def owner : Owner := ⟨.program ⟨257⟩, ⟨38920⟩⟩
def mergeEvent : Nat := 238946
def frameStart : Nat := 238853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩] } }
def leftRaw : List Term := Proof.Events933.exact238941RawTerms
def rightRaw : List Term := Proof.Events933.exact238898RawTerms
def group : MergeGroup := .operator 238941 238898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238941) (leftOrdinal := 1)
    (rightResult := 238898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38917⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238946

namespace LeftMerge238948
def owner : Owner := ⟨.program ⟨257⟩, ⟨38920⟩⟩
def mergeEvent : Nat := 238948
def frameStart : Nat := 238853
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38417⟩⟩] } }
def rhsRaw : List Term := Proof.Events933.exact238895RawTerms
def group : MergeGroup := .relation 238947
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238947) (rhsResult := 238895)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨38917⟩⟩) ⟨38417⟩ 238895) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38417⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], [⟨.program ⟨257⟩, ⟨38417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238948

namespace LeftMerge238956
def owner : Owner := ⟨.program ⟨257⟩, ⟨37414⟩⟩
def mergeEvent : Nat := 238956
def frameStart : Nat := 238853
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37412⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events933.exact238909RawTerms
def rightRaw : List Term := Proof.Events933.exact238952RawTerms
def group : MergeGroup := .operator 238909 238952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 238909) (leftOrdinal := 0)
    (rightResult := 238952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37412⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37412⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238956

namespace LeftMerge238973
def owner : Owner := ⟨.program ⟨257⟩, ⟨37852⟩⟩
def mergeEvent : Nat := 238973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }
def rhsRaw : List Term := Proof.Events933.exact238970RawTerms
def group : MergeGroup := .relation 238972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238972) (rhsResult := 238970)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37849⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238971 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37849⟩⟩]⟩) (none) 238970) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238973

namespace LeftMerge238974
def owner : Owner := ⟨.program ⟨257⟩, ⟨37852⟩⟩
def mergeEvent : Nat := 238974
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩] } }
def rhsRaw : List Term := Proof.Events933.exact238970RawTerms
def group : MergeGroup := .relation 238972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238972) (rhsResult := 238970)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37849⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238971 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37849⟩⟩]⟩) (none) 238970) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7298⟩⟩, ⟨.program ⟨257⟩, ⟨9553⟩⟩, ⟨.program ⟨257⟩, ⟨38917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge238974

namespace LeftMerge238975
def owner : Owner := ⟨.program ⟨257⟩, ⟨37852⟩⟩
def mergeEvent : Nat := 238975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38417⟩⟩] } }
def rhsRaw : List Term := Proof.Events933.exact238970RawTerms
def group : MergeGroup := .relation 238972
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 238972) (rhsResult := 238970)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37849⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 238971 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨37849⟩⟩]⟩) (none) 238970) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38417⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨13851⟩⟩, ⟨.program ⟨257⟩, ⟨37066⟩⟩], [⟨.program ⟨257⟩, ⟨38417⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge238975

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
