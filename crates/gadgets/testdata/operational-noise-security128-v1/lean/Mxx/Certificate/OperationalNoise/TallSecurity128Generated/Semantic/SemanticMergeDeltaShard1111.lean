import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge181238
def owner : Owner := ⟨.program ⟨257⟩, ⟨13330⟩⟩
def mergeEvent : Nat := 181238
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events707.exact181229RawTerms
def rightRaw : List Term := Proof.Events078.exact20116RawTerms
def group : MergeGroup := .operator 181229 20116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181229) (leftOrdinal := 0)
    (rightResult := 20116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181238

namespace LeftMerge181243
def owner : Owner := ⟨.program ⟨257⟩, ⟨28853⟩⟩
def mergeEvent : Nat := 181243
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events707.exact181239RawTerms
def rightRaw : List Term := Proof.Events707.exact181209RawTerms
def group : MergeGroup := .operator 181239 181209
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181239) (leftOrdinal := 1)
    (rightResult := 181209) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181243

namespace LeftMerge181251
def owner : Owner := ⟨.program ⟨257⟩, ⟨30633⟩⟩
def mergeEvent : Nat := 181251
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩] } }
def leftRaw : List Term := Proof.Events707.exact181245RawTerms
def rightRaw : List Term := Proof.Events707.exact181181RawTerms
def group : MergeGroup := .operator 181245 181181
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181245) (leftOrdinal := 1)
    (rightResult := 181181) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30632⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181251

namespace LeftMerge181253
def owner : Owner := ⟨.program ⟨257⟩, ⟨30633⟩⟩
def mergeEvent : Nat := 181253
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30107⟩⟩] } }
def rhsRaw : List Term := Proof.Events707.exact181178RawTerms
def group : MergeGroup := .relation 181252
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181252) (rhsResult := 181178)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30632⟩⟩) ⟨30107⟩ 181178) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30107⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], [⟨.program ⟨257⟩, ⟨30107⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181253

namespace LeftMerge181254
def owner : Owner := ⟨.program ⟨257⟩, ⟨30633⟩⟩
def mergeEvent : Nat := 181254
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩] } }
def leftRaw : List Term := Proof.Events707.exact181245RawTerms
def rightRaw : List Term := Proof.Events707.exact181181RawTerms
def group : MergeGroup := .operator 181245 181181
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181245) (leftOrdinal := 0)
    (rightResult := 181181) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30632⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181254

namespace LeftMerge181268
def owner : Owner := ⟨.program ⟨257⟩, ⟨29562⟩⟩
def mergeEvent : Nat := 181268
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29559⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events708.exact181262RawTerms
def group : MergeGroup := .operator 178370 181262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 181262) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29559⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181268

namespace LeftMerge181347
def owner : Owner := ⟨.program ⟨257⟩, ⟨28847⟩⟩
def mergeEvent : Nat := 181347
def frameStart : Nat := 181317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events708.exact181343RawTerms
def rightRaw : List Term := Proof.Events708.exact181340RawTerms
def group : MergeGroup := .operator 181343 181340
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181343) (leftOrdinal := 0)
    (rightResult := 181340) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13326⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181347

namespace LeftMerge181377
def owner : Owner := ⟨.program ⟨257⟩, ⟨30380⟩⟩
def mergeEvent : Nat := 181377
def frameStart : Nat := 181317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events708.exact181373RawTerms
def rightRaw : List Term := Proof.Events708.exact181371RawTerms
def group : MergeGroup := .operator 181373 181371
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181373) (leftOrdinal := 0)
    (rightResult := 181371) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181377

namespace LeftMerge181400
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def mergeEvent : Nat := 181400
def frameStart : Nat := 181317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events708.exact181396RawTerms
def rightRaw : List Term := Proof.Events708.exact181393RawTerms
def group : MergeGroup := .operator 181396 181393
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181396) (leftOrdinal := 0)
    (rightResult := 181393) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181400

namespace LeftMerge181409
def owner : Owner := ⟨.program ⟨257⟩, ⟨30635⟩⟩
def mergeEvent : Nat := 181409
def frameStart : Nat := 181317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩] } }
def leftRaw : List Term := Proof.Events708.exact181405RawTerms
def rightRaw : List Term := Proof.Events708.exact181362RawTerms
def group : MergeGroup := .operator 181405 181362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181405) (leftOrdinal := 0)
    (rightResult := 181362) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30632⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181409

namespace LeftMerge181410
def owner : Owner := ⟨.program ⟨257⟩, ⟨30635⟩⟩
def mergeEvent : Nat := 181410
def frameStart : Nat := 181317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩] } }
def leftRaw : List Term := Proof.Events708.exact181405RawTerms
def rightRaw : List Term := Proof.Events708.exact181362RawTerms
def group : MergeGroup := .operator 181405 181362
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181405) (leftOrdinal := 1)
    (rightResult := 181362) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30632⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181410

namespace LeftMerge181412
def owner : Owner := ⟨.program ⟨257⟩, ⟨30635⟩⟩
def mergeEvent : Nat := 181412
def frameStart : Nat := 181317
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30107⟩⟩] } }
def rhsRaw : List Term := Proof.Events708.exact181359RawTerms
def group : MergeGroup := .relation 181411
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181411) (rhsResult := 181359)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30632⟩⟩) ⟨30107⟩ 181359) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30107⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], [⟨.program ⟨257⟩, ⟨30107⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181412

namespace LeftMerge181420
def owner : Owner := ⟨.program ⟨257⟩, ⟨29114⟩⟩
def mergeEvent : Nat := 181420
def frameStart : Nat := 181317
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29112⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events708.exact181373RawTerms
def rightRaw : List Term := Proof.Events708.exact181416RawTerms
def group : MergeGroup := .operator 181373 181416
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181373) (leftOrdinal := 0)
    (rightResult := 181416) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29112⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29112⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181420

namespace LeftMerge181437
def owner : Owner := ⟨.program ⟨257⟩, ⟨29562⟩⟩
def mergeEvent : Nat := 181437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events708.exact181434RawTerms
def group : MergeGroup := .relation 181436
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181436) (rhsResult := 181434)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 181435 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29559⟩⟩]⟩) (none) 181434) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181437

namespace LeftMerge181438
def owner : Owner := ⟨.program ⟨257⟩, ⟨29562⟩⟩
def mergeEvent : Nat := 181438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩] } }
def rhsRaw : List Term := Proof.Events708.exact181434RawTerms
def group : MergeGroup := .relation 181436
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181436) (rhsResult := 181434)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 181435 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29559⟩⟩]⟩) (none) 181434) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30632⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181438

namespace LeftMerge181439
def owner : Owner := ⟨.program ⟨257⟩, ⟨29562⟩⟩
def mergeEvent : Nat := 181439
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30107⟩⟩] } }
def rhsRaw : List Term := Proof.Events708.exact181434RawTerms
def group : MergeGroup := .relation 181436
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 181436) (rhsResult := 181434)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29559⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 181435 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29559⟩⟩]⟩) (none) 181434) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30107⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13326⟩⟩, ⟨.program ⟨257⟩, ⟨28846⟩⟩], [⟨.program ⟨257⟩, ⟨30107⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181439

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
