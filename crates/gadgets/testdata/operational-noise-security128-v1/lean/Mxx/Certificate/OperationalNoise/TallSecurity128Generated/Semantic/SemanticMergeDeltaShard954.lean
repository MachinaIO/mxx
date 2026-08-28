import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge156917
def owner : Owner := ⟨.program ⟨257⟩, ⟨18203⟩⟩
def mergeEvent : Nat := 156917
def frameStart : Nat := 156887
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events612.exact156913RawTerms
def rightRaw : List Term := Proof.Events612.exact156910RawTerms
def group : MergeGroup := .operator 156913 156910
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156913) (leftOrdinal := 0)
    (rightResult := 156910) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12636⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156917

namespace LeftMerge156947
def owner : Owner := ⟨.program ⟨257⟩, ⟨19976⟩⟩
def mergeEvent : Nat := 156947
def frameStart : Nat := 156887
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events613.exact156943RawTerms
def rightRaw : List Term := Proof.Events613.exact156941RawTerms
def group : MergeGroup := .operator 156943 156941
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156943) (leftOrdinal := 0)
    (rightResult := 156941) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156947

namespace LeftMerge156970
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def mergeEvent : Nat := 156970
def frameStart : Nat := 156887
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }
def leftRaw : List Term := Proof.Events613.exact156966RawTerms
def rightRaw : List Term := Proof.Events613.exact156963RawTerms
def group : MergeGroup := .operator 156966 156963
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156966) (leftOrdinal := 0)
    (rightResult := 156963) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9571⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156970

namespace LeftMerge156979
def owner : Owner := ⟨.program ⟨257⟩, ⟨20189⟩⟩
def mergeEvent : Nat := 156979
def frameStart : Nat := 156887
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩] } }
def leftRaw : List Term := Proof.Events613.exact156975RawTerms
def rightRaw : List Term := Proof.Events613.exact156932RawTerms
def group : MergeGroup := .operator 156975 156932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156975) (leftOrdinal := 0)
    (rightResult := 156932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20186⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156979

namespace LeftMerge156980
def owner : Owner := ⟨.program ⟨257⟩, ⟨20189⟩⟩
def mergeEvent : Nat := 156980
def frameStart : Nat := 156887
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩] } }
def leftRaw : List Term := Proof.Events613.exact156975RawTerms
def rightRaw : List Term := Proof.Events613.exact156932RawTerms
def group : MergeGroup := .operator 156975 156932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156975) (leftOrdinal := 1)
    (rightResult := 156932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20186⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156980

namespace LeftMerge156982
def owner : Owner := ⟨.program ⟨257⟩, ⟨20189⟩⟩
def mergeEvent : Nat := 156982
def frameStart : Nat := 156887
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19691⟩⟩] } }
def rhsRaw : List Term := Proof.Events613.exact156929RawTerms
def group : MergeGroup := .relation 156981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 156981) (rhsResult := 156929)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20186⟩⟩) ⟨19691⟩ 156929) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19691⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], [⟨.program ⟨257⟩, ⟨19691⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge156982

namespace LeftMerge156990
def owner : Owner := ⟨.program ⟨257⟩, ⟨18566⟩⟩
def mergeEvent : Nat := 156990
def frameStart : Nat := 156887
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events613.exact156943RawTerms
def rightRaw : List Term := Proof.Events613.exact156986RawTerms
def group : MergeGroup := .operator 156943 156986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 156943) (leftOrdinal := 0)
    (rightResult := 156986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18564⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge156990

namespace LeftMerge157007
def owner : Owner := ⟨.program ⟨257⟩, ⟨19122⟩⟩
def mergeEvent : Nat := 157007
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }
def rhsRaw : List Term := Proof.Events613.exact157004RawTerms
def group : MergeGroup := .relation 157006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157006) (rhsResult := 157004)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19119⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 157005 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19119⟩⟩]⟩) (none) 157004) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157007

namespace LeftMerge157008
def owner : Owner := ⟨.program ⟨257⟩, ⟨19122⟩⟩
def mergeEvent : Nat := 157008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩] } }
def rhsRaw : List Term := Proof.Events613.exact157004RawTerms
def group : MergeGroup := .relation 157006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157006) (rhsResult := 157004)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19119⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 157005 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19119⟩⟩]⟩) (none) 157004) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157008

namespace LeftMerge157009
def owner : Owner := ⟨.program ⟨257⟩, ⟨19122⟩⟩
def mergeEvent : Nat := 157009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19691⟩⟩] } }
def rhsRaw : List Term := Proof.Events613.exact157004RawTerms
def group : MergeGroup := .relation 157006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157006) (rhsResult := 157004)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19119⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 157005 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19119⟩⟩]⟩) (none) 157004) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19691⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], [⟨.program ⟨257⟩, ⟨19691⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157009

namespace LeftMerge157010
def owner : Owner := ⟨.program ⟨257⟩, ⟨19122⟩⟩
def mergeEvent : Nat := 157010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events613.exact157004RawTerms
def group : MergeGroup := .relation 157006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157006) (rhsResult := 157004)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19119⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 157005 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19119⟩⟩]⟩) (none) 157004) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157010

namespace LeftMerge157015
def owner : Owner := ⟨.program ⟨257⟩, ⟨20188⟩⟩
def mergeEvent : Nat := 157015
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19691⟩⟩] } }
def leftRaw : List Term := Proof.Events613.exact157011RawTerms
def rightRaw : List Term := Proof.Events612.exact156825RawTerms
def group : MergeGroup := .operator 157011 156825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157011) (leftOrdinal := 2)
    (rightResult := 156825) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19691⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19691⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨12636⟩⟩, ⟨.program ⟨257⟩, ⟨18202⟩⟩], [⟨.program ⟨257⟩, ⟨19691⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157015

namespace LeftMerge157016
def owner : Owner := ⟨.program ⟨257⟩, ⟨20188⟩⟩
def mergeEvent : Nat := 157016
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩] } }
def leftRaw : List Term := Proof.Events613.exact157011RawTerms
def rightRaw : List Term := Proof.Events612.exact156825RawTerms
def group : MergeGroup := .operator 157011 156825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157011) (leftOrdinal := 1)
    (rightResult := 156825) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20186⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157016

namespace LeftMerge157024
def owner : Owner := ⟨.program ⟨257⟩, ⟨20561⟩⟩
def mergeEvent : Nat := 157024
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20559⟩⟩] } }
def leftRaw : List Term := Proof.Events613.exact157018RawTerms
def rightRaw : List Term := Proof.Events612.exact156741RawTerms
def group : MergeGroup := .operator 157018 156741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157018) (leftOrdinal := 0)
    (rightResult := 156741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20559⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20559⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge157024

namespace LeftMerge157025
def owner : Owner := ⟨.program ⟨257⟩, ⟨20561⟩⟩
def mergeEvent : Nat := 157025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20559⟩⟩] } }
def leftRaw : List Term := Proof.Events613.exact157018RawTerms
def rightRaw : List Term := Proof.Events612.exact156741RawTerms
def group : MergeGroup := .operator 157018 156741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 157018) (leftOrdinal := 1)
    (rightResult := 156741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20559⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20559⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157025

namespace LeftMerge157027
def owner : Owner := ⟨.program ⟨257⟩, ⟨20561⟩⟩
def mergeEvent : Nat := 157027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19834⟩⟩] } }
def rhsRaw : List Term := Proof.Events612.exact156738RawTerms
def group : MergeGroup := .relation 157026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 157026) (rhsResult := 156738)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20559⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20559⟩⟩) ⟨19834⟩ 156738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19834⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨18564⟩⟩], [⟨.program ⟨257⟩, ⟨19834⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge157027

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
