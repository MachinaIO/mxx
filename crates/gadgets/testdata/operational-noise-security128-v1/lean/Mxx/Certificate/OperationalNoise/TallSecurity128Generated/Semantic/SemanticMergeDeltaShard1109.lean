import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge180928
def owner : Owner := ⟨.program ⟨257⟩, ⟨36295⟩⟩
def mergeEvent : Nat := 180928
def frameStart : Nat := 180835
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩] } }
def leftRaw : List Term := Proof.Events706.exact180923RawTerms
def rightRaw : List Term := Proof.Events706.exact180880RawTerms
def group : MergeGroup := .operator 180923 180880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180923) (leftOrdinal := 1)
    (rightResult := 180880) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36292⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180928

namespace LeftMerge180930
def owner : Owner := ⟨.program ⟨257⟩, ⟨36295⟩⟩
def mergeEvent : Nat := 180930
def frameStart : Nat := 180835
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35767⟩⟩] } }
def rhsRaw : List Term := Proof.Events706.exact180877RawTerms
def group : MergeGroup := .relation 180929
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180929) (rhsResult := 180877)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36292⟩⟩) ⟨35767⟩ 180877) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35767⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], [⟨.program ⟨257⟩, ⟨35767⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180930

namespace LeftMerge180938
def owner : Owner := ⟨.program ⟨257⟩, ⟨34774⟩⟩
def mergeEvent : Nat := 180938
def frameStart : Nat := 180835
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events706.exact180891RawTerms
def rightRaw : List Term := Proof.Events706.exact180934RawTerms
def group : MergeGroup := .operator 180891 180934
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180891) (leftOrdinal := 0)
    (rightResult := 180934) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180938

namespace LeftMerge180955
def owner : Owner := ⟨.program ⟨257⟩, ⟨35222⟩⟩
def mergeEvent : Nat := 180955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events706.exact180952RawTerms
def group : MergeGroup := .relation 180954
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180954) (rhsResult := 180952)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 180953 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35219⟩⟩]⟩) (none) 180952) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180955

namespace LeftMerge180956
def owner : Owner := ⟨.program ⟨257⟩, ⟨35222⟩⟩
def mergeEvent : Nat := 180956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩] } }
def rhsRaw : List Term := Proof.Events706.exact180952RawTerms
def group : MergeGroup := .relation 180954
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180954) (rhsResult := 180952)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 180953 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35219⟩⟩]⟩) (none) 180952) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180956

namespace LeftMerge180957
def owner : Owner := ⟨.program ⟨257⟩, ⟨35222⟩⟩
def mergeEvent : Nat := 180957
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35767⟩⟩] } }
def rhsRaw : List Term := Proof.Events706.exact180952RawTerms
def group : MergeGroup := .relation 180954
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180954) (rhsResult := 180952)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 180953 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35219⟩⟩]⟩) (none) 180952) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35767⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], [⟨.program ⟨257⟩, ⟨35767⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180957

namespace LeftMerge180958
def owner : Owner := ⟨.program ⟨257⟩, ⟨35222⟩⟩
def mergeEvent : Nat := 180958
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events706.exact180952RawTerms
def group : MergeGroup := .relation 180954
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180954) (rhsResult := 180952)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35219⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 180953 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35219⟩⟩]⟩) (none) 180952) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180958

namespace LeftMerge180963
def owner : Owner := ⟨.program ⟨257⟩, ⟨36294⟩⟩
def mergeEvent : Nat := 180963
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35767⟩⟩] } }
def leftRaw : List Term := Proof.Events706.exact180959RawTerms
def rightRaw : List Term := Proof.Events706.exact180773RawTerms
def group : MergeGroup := .operator 180959 180773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180959) (leftOrdinal := 2)
    (rightResult := 180773) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35767⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35767⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨13626⟩⟩, ⟨.program ⟨257⟩, ⟨34506⟩⟩], [⟨.program ⟨257⟩, ⟨35767⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180963

namespace LeftMerge180964
def owner : Owner := ⟨.program ⟨257⟩, ⟨36294⟩⟩
def mergeEvent : Nat := 180964
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩] } }
def leftRaw : List Term := Proof.Events706.exact180959RawTerms
def rightRaw : List Term := Proof.Events706.exact180773RawTerms
def group : MergeGroup := .operator 180959 180773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180959) (leftOrdinal := 1)
    (rightResult := 180773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36292⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180964

namespace LeftMerge180972
def owner : Owner := ⟨.program ⟨257⟩, ⟨36706⟩⟩
def mergeEvent : Nat := 180972
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36704⟩⟩] } }
def leftRaw : List Term := Proof.Events706.exact180966RawTerms
def rightRaw : List Term := Proof.Events705.exact180689RawTerms
def group : MergeGroup := .operator 180966 180689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180966) (leftOrdinal := 0)
    (rightResult := 180689) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36704⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36704⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180972

namespace LeftMerge180973
def owner : Owner := ⟨.program ⟨257⟩, ⟨36706⟩⟩
def mergeEvent : Nat := 180973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36704⟩⟩] } }
def leftRaw : List Term := Proof.Events706.exact180966RawTerms
def rightRaw : List Term := Proof.Events705.exact180689RawTerms
def group : MergeGroup := .operator 180966 180689
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 180966) (leftOrdinal := 1)
    (rightResult := 180689) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36704⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36704⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180973

namespace LeftMerge180975
def owner : Owner := ⟨.program ⟨257⟩, ⟨36706⟩⟩
def mergeEvent : Nat := 180975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35928⟩⟩] } }
def rhsRaw : List Term := Proof.Events705.exact180686RawTerms
def group : MergeGroup := .relation 180974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 180974) (rhsResult := 180686)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36704⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36704⟩⟩) ⟨35928⟩ 180686) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35928⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩, ⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨35928⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge180975

namespace LeftMerge180989
def owner : Owner := ⟨.program ⟨257⟩, ⟨35559⟩⟩
def mergeEvent : Nat := 180989
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35556⟩⟩] } }
def leftRaw : List Term := Proof.Events696.exact178370RawTerms
def rightRaw : List Term := Proof.Events706.exact180983RawTerms
def group : MergeGroup := .operator 178370 180983
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 178370) (leftOrdinal := 0)
    (rightResult := 180983) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35556⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6452⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35556⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge180989

namespace LeftMerge181110
def owner : Owner := ⟨.program ⟨257⟩, ⟨36120⟩⟩
def mergeEvent : Nat := 181110
def frameStart : Nat := 181044
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events707.exact181106RawTerms
def rightRaw : List Term := Proof.Events707.exact181104RawTerms
def group : MergeGroup := .operator 181106 181104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181106) (leftOrdinal := 0)
    (rightResult := 181104) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181110

namespace LeftMerge181122
def owner : Owner := ⟨.program ⟨257⟩, ⟨36705⟩⟩
def mergeEvent : Nat := 181122
def frameStart : Nat := 181044
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36704⟩⟩] } }
def leftRaw : List Term := Proof.Events707.exact181118RawTerms
def rightRaw : List Term := Proof.Events707.exact181095RawTerms
def group : MergeGroup := .operator 181118 181095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181118) (leftOrdinal := 0)
    (rightResult := 181095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36704⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36704⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge181122

namespace LeftMerge181123
def owner : Owner := ⟨.program ⟨257⟩, ⟨36705⟩⟩
def mergeEvent : Nat := 181123
def frameStart : Nat := 181044
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36704⟩⟩] } }
def leftRaw : List Term := Proof.Events707.exact181118RawTerms
def rightRaw : List Term := Proof.Events707.exact181095RawTerms
def group : MergeGroup := .operator 181118 181095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 181118) (leftOrdinal := 1)
    (rightResult := 181095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34772⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36704⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34772⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36704⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge181123

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
