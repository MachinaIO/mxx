import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge103747
def owner : Owner := ⟨.program ⟨257⟩, ⟨53101⟩⟩
def mergeEvent : Nat := 103747
def frameStart : Nat := 103666
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52205⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103714RawTerms
def group : MergeGroup := .relation 103746
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103746) (rhsResult := 103714)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53100⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53100⟩⟩) ⟨52205⟩ 103714) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52205⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52205⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103747

namespace LeftMerge103755
def owner : Owner := ⟨.program ⟨257⟩, ⟨51263⟩⟩
def mergeEvent : Nat := 103755
def frameStart : Nat := 103666
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103728RawTerms
def rightRaw : List Term := Proof.Events405.exact103751RawTerms
def group : MergeGroup := .operator 103728 103751
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103728) (leftOrdinal := 0)
    (rightResult := 103751) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51260⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103755

namespace LeftMerge103772
def owner : Owner := ⟨.program ⟨257⟩, ⟨51855⟩⟩
def mergeEvent : Nat := 103772
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103769RawTerms
def group : MergeGroup := .relation 103771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103771) (rhsResult := 103769)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51852⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103770 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51852⟩⟩]⟩) (none) 103769) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103772

namespace LeftMerge103773
def owner : Owner := ⟨.program ⟨257⟩, ⟨51855⟩⟩
def mergeEvent : Nat := 103773
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53100⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103769RawTerms
def group : MergeGroup := .relation 103771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103771) (rhsResult := 103769)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51852⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103770 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51852⟩⟩]⟩) (none) 103769) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53100⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53100⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103773

namespace LeftMerge103774
def owner : Owner := ⟨.program ⟨257⟩, ⟨51855⟩⟩
def mergeEvent : Nat := 103774
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52205⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103769RawTerms
def group : MergeGroup := .relation 103771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103771) (rhsResult := 103769)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51852⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103770 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51852⟩⟩]⟩) (none) 103769) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52205⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52205⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103774

namespace LeftMerge103775
def owner : Owner := ⟨.program ⟨257⟩, ⟨51855⟩⟩
def mergeEvent : Nat := 103775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103769RawTerms
def group : MergeGroup := .relation 103771
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103771) (rhsResult := 103769)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51852⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 103770 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51852⟩⟩]⟩) (none) 103769) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103775

namespace LeftMerge103780
def owner : Owner := ⟨.program ⟨257⟩, ⟨53103⟩⟩
def mergeEvent : Nat := 103780
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53100⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103776RawTerms
def rightRaw : List Term := Proof.Events404.exact103598RawTerms
def group : MergeGroup := .operator 103776 103598
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103776) (leftOrdinal := 0)
    (rightResult := 103598) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53100⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53100⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53100⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103780

namespace LeftMerge103781
def owner : Owner := ⟨.program ⟨257⟩, ⟨53103⟩⟩
def mergeEvent : Nat := 103781
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52205⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103776RawTerms
def rightRaw : List Term := Proof.Events404.exact103598RawTerms
def group : MergeGroup := .operator 103776 103598
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103776) (leftOrdinal := 2)
    (rightResult := 103598) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52205⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52205⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52205⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103781

namespace LeftMerge103789
def owner : Owner := ⟨.program ⟨257⟩, ⟨53104⟩⟩
def mergeEvent : Nat := 103789
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103783RawTerms
def rightRaw : List Term := Proof.Events061.exact15802RawTerms
def group : MergeGroup := .operator 103783 15802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103783) (leftOrdinal := 0)
    (rightResult := 15802) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103789

namespace LeftMerge103790
def owner : Owner := ⟨.program ⟨257⟩, ⟨53104⟩⟩
def mergeEvent : Nat := 103790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }
def leftRaw : List Term := Proof.Events405.exact103783RawTerms
def rightRaw : List Term := Proof.Events061.exact15802RawTerms
def group : MergeGroup := .operator 103783 15802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103783) (leftOrdinal := 1)
    (rightResult := 15802) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103790

namespace LeftMerge103792
def owner : Owner := ⟨.program ⟨257⟩, ⟨53104⟩⟩
def mergeEvent : Nat := 103792
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15795RawTerms
def group : MergeGroup := .relation 103791
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103791) (rhsResult := 15795)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7131⟩⟩) ⟨7031⟩ 15795) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨51260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103792

namespace LeftMerge103806
def owner : Owner := ⟨.program ⟨257⟩, ⟨34042⟩⟩
def mergeEvent : Nat := 103806
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34040⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97554RawTerms
def rightRaw : List Term := Proof.Events405.exact103800RawTerms
def group : MergeGroup := .operator 97554 103800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97554) (leftOrdinal := 0)
    (rightResult := 103800) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34040⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨34040⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103806

namespace LeftMerge103807
def owner : Owner := ⟨.program ⟨257⟩, ⟨34042⟩⟩
def mergeEvent : Nat := 103807
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34040⟩⟩] } }
def leftRaw : List Term := Proof.Events381.exact97554RawTerms
def rightRaw : List Term := Proof.Events405.exact103800RawTerms
def group : MergeGroup := .operator 97554 103800
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97554) (leftOrdinal := 1)
    (rightResult := 103800) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨34040⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34040⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103807

namespace LeftMerge103809
def owner : Owner := ⟨.program ⟨257⟩, ⟨34042⟩⟩
def mergeEvent : Nat := 103809
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33145⟩⟩] } }
def rhsRaw : List Term := Proof.Events405.exact103797RawTerms
def group : MergeGroup := .relation 103808
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 103808) (rhsResult := 103797)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨34040⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨34040⟩⟩) ⟨33145⟩ 103797) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33145⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨33145⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge103809

namespace LeftMerge103823
def owner : Owner := ⟨.program ⟨257⟩, ⟨32795⟩⟩
def mergeEvent : Nat := 103823
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32792⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events405.exact103817RawTerms
def group : MergeGroup := .operator 90620 103817
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 103817) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32792⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32792⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103823

namespace LeftMerge103944
def owner : Owner := ⟨.program ⟨257⟩, ⟨33328⟩⟩
def mergeEvent : Nat := 103944
def frameStart : Nat := 103878
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events406.exact103940RawTerms
def rightRaw : List Term := Proof.Events406.exact103938RawTerms
def group : MergeGroup := .operator 103940 103938
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 103940) (leftOrdinal := 0)
    (rightResult := 103938) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31868⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31868⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge103944

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
