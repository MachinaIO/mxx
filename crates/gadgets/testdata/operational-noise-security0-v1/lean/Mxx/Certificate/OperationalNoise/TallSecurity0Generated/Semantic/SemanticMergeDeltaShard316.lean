import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge51811
def owner : Owner := ⟨.program ⟨214⟩, ⟨12967⟩⟩
def mergeEvent : Nat := 51811
def frameStart : Nat := 51781
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events202.exact51807RawTerms
def rightRaw : List Term := Proof.Events202.exact51804RawTerms
def group : MergeGroup := .operator 51807 51804
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51807) (leftOrdinal := 0)
    (rightResult := 51804) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51811

namespace LeftMerge51841
def owner : Owner := ⟨.program ⟨214⟩, ⟨13060⟩⟩
def mergeEvent : Nat := 51841
def frameStart : Nat := 51781
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events202.exact51837RawTerms
def rightRaw : List Term := Proof.Events202.exact51835RawTerms
def group : MergeGroup := .operator 51837 51835
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51837) (leftOrdinal := 0)
    (rightResult := 51835) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51841

namespace LeftMerge51864
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def mergeEvent : Nat := 51864
def frameStart : Nat := 51781
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }
def leftRaw : List Term := Proof.Events202.exact51860RawTerms
def rightRaw : List Term := Proof.Events202.exact51857RawTerms
def group : MergeGroup := .operator 51860 51857
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51860) (leftOrdinal := 0)
    (rightResult := 51857) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7876⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51864

namespace LeftMerge51873
def owner : Owner := ⟨.program ⟨214⟩, ⟨25612⟩⟩
def mergeEvent : Nat := 51873
def frameStart : Nat := 51781
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩] } }
def leftRaw : List Term := Proof.Events202.exact51869RawTerms
def rightRaw : List Term := Proof.Events202.exact51826RawTerms
def group : MergeGroup := .operator 51869 51826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51869) (leftOrdinal := 0)
    (rightResult := 51826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25609⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51873

namespace LeftMerge51874
def owner : Owner := ⟨.program ⟨214⟩, ⟨25612⟩⟩
def mergeEvent : Nat := 51874
def frameStart : Nat := 51781
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩] } }
def leftRaw : List Term := Proof.Events202.exact51869RawTerms
def rightRaw : List Term := Proof.Events202.exact51826RawTerms
def group : MergeGroup := .operator 51869 51826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51869) (leftOrdinal := 1)
    (rightResult := 51826) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25609⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51874

namespace LeftMerge51876
def owner : Owner := ⟨.program ⟨214⟩, ⟨25612⟩⟩
def mergeEvent : Nat := 51876
def frameStart : Nat := 51781
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23334⟩⟩] } }
def rhsRaw : List Term := Proof.Events202.exact51823RawTerms
def group : MergeGroup := .relation 51875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51875) (rhsResult := 51823)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25609⟩⟩) ⟨23334⟩ 51823) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23334⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨23334⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51876

namespace LeftMerge51884
def owner : Owner := ⟨.program ⟨214⟩, ⟨16758⟩⟩
def mergeEvent : Nat := 51884
def frameStart : Nat := 51781
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events202.exact51837RawTerms
def rightRaw : List Term := Proof.Events202.exact51880RawTerms
def group : MergeGroup := .operator 51837 51880
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51837) (leftOrdinal := 0)
    (rightResult := 51880) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16756⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51884

namespace LeftMerge51901
def owner : Owner := ⟨.program ⟨214⟩, ⟨20111⟩⟩
def mergeEvent : Nat := 51901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }
def rhsRaw : List Term := Proof.Events202.exact51898RawTerms
def group : MergeGroup := .relation 51900
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51900) (rhsResult := 51898)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51899 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩) (none) 51898) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51901

namespace LeftMerge51902
def owner : Owner := ⟨.program ⟨214⟩, ⟨20111⟩⟩
def mergeEvent : Nat := 51902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩] } }
def rhsRaw : List Term := Proof.Events202.exact51898RawTerms
def group : MergeGroup := .relation 51900
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51900) (rhsResult := 51898)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51899 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩) (none) 51898) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51902

namespace LeftMerge51903
def owner : Owner := ⟨.program ⟨214⟩, ⟨20111⟩⟩
def mergeEvent : Nat := 51903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23334⟩⟩] } }
def rhsRaw : List Term := Proof.Events202.exact51898RawTerms
def group : MergeGroup := .relation 51900
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51900) (rhsResult := 51898)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51899 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩) (none) 51898) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23334⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨23334⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51903

namespace LeftMerge51904
def owner : Owner := ⟨.program ⟨214⟩, ⟨20111⟩⟩
def mergeEvent : Nat := 51904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events202.exact51898RawTerms
def group : MergeGroup := .relation 51900
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51900) (rhsResult := 51898)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51899 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20108⟩⟩]⟩) (none) 51898) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51904

namespace LeftMerge51909
def owner : Owner := ⟨.program ⟨214⟩, ⟨25611⟩⟩
def mergeEvent : Nat := 51909
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23334⟩⟩] } }
def leftRaw : List Term := Proof.Events202.exact51905RawTerms
def rightRaw : List Term := Proof.Events202.exact51719RawTerms
def group : MergeGroup := .operator 51905 51719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51905) (leftOrdinal := 2)
    (rightResult := 51719) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23334⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23334⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨10140⟩⟩, ⟨.program ⟨214⟩, ⟨12966⟩⟩], [⟨.program ⟨214⟩, ⟨23334⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51909

namespace LeftMerge51910
def owner : Owner := ⟨.program ⟨214⟩, ⟨25611⟩⟩
def mergeEvent : Nat := 51910
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩] } }
def leftRaw : List Term := Proof.Events202.exact51905RawTerms
def rightRaw : List Term := Proof.Events202.exact51719RawTerms
def group : MergeGroup := .operator 51905 51719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51905) (leftOrdinal := 1)
    (rightResult := 51719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6768⟩⟩, ⟨.program ⟨214⟩, ⟨7876⟩⟩, ⟨.program ⟨214⟩, ⟨25609⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51910

namespace LeftMerge51918
def owner : Owner := ⟨.program ⟨214⟩, ⟨29617⟩⟩
def mergeEvent : Nat := 51918
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29615⟩⟩] } }
def leftRaw : List Term := Proof.Events202.exact51912RawTerms
def rightRaw : List Term := Proof.Events201.exact51635RawTerms
def group : MergeGroup := .operator 51912 51635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51912) (leftOrdinal := 0)
    (rightResult := 51635) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6705⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29615⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6705⟩⟩, ⟨.program ⟨214⟩, ⟨29615⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51918

namespace LeftMerge51919
def owner : Owner := ⟨.program ⟨214⟩, ⟨29617⟩⟩
def mergeEvent : Nat := 51919
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29615⟩⟩] } }
def leftRaw : List Term := Proof.Events202.exact51912RawTerms
def rightRaw : List Term := Proof.Events201.exact51635RawTerms
def group : MergeGroup := .operator 51912 51635
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51912) (leftOrdinal := 1)
    (rightResult := 51635) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29615⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29615⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51919

namespace LeftMerge51921
def owner : Owner := ⟨.program ⟨214⟩, ⟨29617⟩⟩
def mergeEvent : Nat := 51921
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24669⟩⟩] } }
def rhsRaw : List Term := Proof.Events201.exact51632RawTerms
def group : MergeGroup := .relation 51920
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51920) (rhsResult := 51632)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29615⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29615⟩⟩) ⟨24669⟩ 51632) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24669⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨16756⟩⟩], [⟨.program ⟨214⟩, ⟨24669⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51921

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
