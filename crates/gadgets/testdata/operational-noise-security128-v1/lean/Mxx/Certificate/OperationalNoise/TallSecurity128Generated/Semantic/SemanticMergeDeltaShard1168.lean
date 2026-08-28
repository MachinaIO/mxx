import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge191106
def owner : Owner := ⟨.program ⟨257⟩, ⟨59001⟩⟩
def mergeEvent : Nat := 191106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58998⟩⟩] } }
def leftRaw : List Term := Proof.Events746.exact191102RawTerms
def rightRaw : List Term := Proof.Events745.exact190924RawTerms
def group : MergeGroup := .operator 191102 190924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 191102) (leftOrdinal := 0)
    (rightResult := 190924) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58998⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58998⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7185⟩⟩, ⟨.program ⟨257⟩, ⟨58998⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge191106

namespace LeftMerge191107
def owner : Owner := ⟨.program ⟨257⟩, ⟨59001⟩⟩
def mergeEvent : Nat := 191107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58147⟩⟩] } }
def leftRaw : List Term := Proof.Events746.exact191102RawTerms
def rightRaw : List Term := Proof.Events745.exact190924RawTerms
def group : MergeGroup := .operator 191102 190924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 191102) (leftOrdinal := 2)
    (rightResult := 190924) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58147⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨58147⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨56872⟩⟩], [⟨.program ⟨257⟩, ⟨58147⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge191107

namespace LeftMerge191115
def owner : Owner := ⟨.program ⟨257⟩, ⟨59002⟩⟩
def mergeEvent : Nat := 191115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩] } }
def leftRaw : List Term := Proof.Events746.exact191109RawTerms
def rightRaw : List Term := Proof.Events061.exact15762RawTerms
def group : MergeGroup := .operator 191109 15762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 191109) (leftOrdinal := 0)
    (rightResult := 15762) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7209⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7107⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7209⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge191115

namespace LeftMerge191116
def owner : Owner := ⟨.program ⟨257⟩, ⟨59002⟩⟩
def mergeEvent : Nat := 191116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩] } }
def leftRaw : List Term := Proof.Events746.exact191109RawTerms
def rightRaw : List Term := Proof.Events061.exact15762RawTerms
def group : MergeGroup := .operator 191109 15762
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 191109) (leftOrdinal := 1)
    (rightResult := 15762) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7107⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge191116

namespace LeftMerge191118
def owner : Owner := ⟨.program ⟨257⟩, ⟨59002⟩⟩
def mergeEvent : Nat := 191118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15755RawTerms
def group : MergeGroup := .relation 191117
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 191117) (rhsResult := 15755)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7107⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7107⟩⟩) ⟨7019⟩ 15755) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57182⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge191118

namespace LeftMerge191132
def owner : Owner := ⟨.program ⟨257⟩, ⟨56020⟩⟩
def mergeEvent : Nat := 191132
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184340RawTerms
def rightRaw : List Term := Proof.Events746.exact191126RawTerms
def group : MergeGroup := .operator 184340 191126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184340) (leftOrdinal := 0)
    (rightResult := 191126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56018⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge191132

namespace LeftMerge191133
def owner : Owner := ⟨.program ⟨257⟩, ⟨56020⟩⟩
def mergeEvent : Nat := 191133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩] } }
def leftRaw : List Term := Proof.Events720.exact184340RawTerms
def rightRaw : List Term := Proof.Events746.exact191126RawTerms
def group : MergeGroup := .operator 184340 191126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 184340) (leftOrdinal := 1)
    (rightResult := 191126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56018⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge191133

namespace LeftMerge191135
def owner : Owner := ⟨.program ⟨257⟩, ⟨56020⟩⟩
def mergeEvent : Nat := 191135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55167⟩⟩] } }
def rhsRaw : List Term := Proof.Events746.exact191123RawTerms
def group : MergeGroup := .relation 191134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 191134) (rhsResult := 191123)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56018⟩⟩) ⟨55167⟩ 191123) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55167⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨55167⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge191135

namespace LeftMerge191149
def owner : Owner := ⟨.program ⟨257⟩, ⟨54795⟩⟩
def mergeEvent : Nat := 191149
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54792⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events746.exact191143RawTerms
def group : MergeGroup := .operator 178370 191143
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 191143) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54792⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54792⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge191149

namespace LeftMerge191270
def owner : Owner := ⟨.program ⟨257⟩, ⟨55360⟩⟩
def mergeEvent : Nat := 191270
def frameStart : Nat := 191204
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events747.exact191266RawTerms
def rightRaw : List Term := Proof.Events747.exact191264RawTerms
def group : MergeGroup := .operator 191266 191264
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 191266) (leftOrdinal := 0)
    (rightResult := 191264) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge191270

namespace LeftMerge191282
def owner : Owner := ⟨.program ⟨257⟩, ⟨56019⟩⟩
def mergeEvent : Nat := 191282
def frameStart : Nat := 191204
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩] } }
def leftRaw : List Term := Proof.Events747.exact191278RawTerms
def rightRaw : List Term := Proof.Events747.exact191255RawTerms
def group : MergeGroup := .operator 191278 191255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 191278) (leftOrdinal := 0)
    (rightResult := 191255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56018⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge191282

namespace LeftMerge191283
def owner : Owner := ⟨.program ⟨257⟩, ⟨56019⟩⟩
def mergeEvent : Nat := 191283
def frameStart : Nat := 191204
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩] } }
def leftRaw : List Term := Proof.Events747.exact191278RawTerms
def rightRaw : List Term := Proof.Events747.exact191255RawTerms
def group : MergeGroup := .operator 191278 191255
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 191278) (leftOrdinal := 1)
    (rightResult := 191255) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨56018⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge191283

namespace LeftMerge191285
def owner : Owner := ⟨.program ⟨257⟩, ⟨56019⟩⟩
def mergeEvent : Nat := 191285
def frameStart : Nat := 191204
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53892⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55167⟩⟩] } }
def rhsRaw : List Term := Proof.Events747.exact191252RawTerms
def group : MergeGroup := .relation 191284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 191284) (rhsResult := 191252)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨56018⟩⟩) ⟨55167⟩ 191252) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55167⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53892⟩⟩], [⟨.program ⟨257⟩, ⟨55167⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge191285

namespace LeftMerge191293
def owner : Owner := ⟨.program ⟨257⟩, ⟨54205⟩⟩
def mergeEvent : Nat := 191293
def frameStart : Nat := 191204
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events747.exact191266RawTerms
def rightRaw : List Term := Proof.Events747.exact191289RawTerms
def group : MergeGroup := .operator 191266 191289
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 191266) (leftOrdinal := 0)
    (rightResult := 191289) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54202⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge191293

namespace LeftMerge191310
def owner : Owner := ⟨.program ⟨257⟩, ⟨54795⟩⟩
def mergeEvent : Nat := 191310
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }
def rhsRaw : List Term := Proof.Events747.exact191307RawTerms
def group : MergeGroup := .relation 191309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 191309) (rhsResult := 191307)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54792⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 191308 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54792⟩⟩]⟩) (none) 191307) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge191310

namespace LeftMerge191311
def owner : Owner := ⟨.program ⟨257⟩, ⟨54795⟩⟩
def mergeEvent : Nat := 191311
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩] } }
def rhsRaw : List Term := Proof.Events747.exact191307RawTerms
def group : MergeGroup := .relation 191309
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 191309) (rhsResult := 191307)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54792⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 191308 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54792⟩⟩]⟩) (none) 191307) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨56018⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge191311

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
