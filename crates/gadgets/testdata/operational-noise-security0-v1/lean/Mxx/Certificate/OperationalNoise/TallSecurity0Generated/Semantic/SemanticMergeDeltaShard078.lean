import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge14760
def owner : Owner := ⟨.program ⟨214⟩, ⟨25010⟩⟩
def mergeEvent : Nat := 14760
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23004⟩⟩] } }
def leftRaw : List Term := Proof.Events057.exact14756RawTerms
def rightRaw : List Term := Proof.Events056.exact14570RawTerms
def group : MergeGroup := .operator 14756 14570
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14756) (leftOrdinal := 2)
    (rightResult := 14570) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23004⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23004⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨9525⟩⟩, ⟨.program ⟨214⟩, ⟨10708⟩⟩], [⟨.program ⟨214⟩, ⟨23004⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14760

namespace LeftMerge14761
def owner : Owner := ⟨.program ⟨214⟩, ⟨25010⟩⟩
def mergeEvent : Nat := 14761
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩] } }
def leftRaw : List Term := Proof.Events057.exact14756RawTerms
def rightRaw : List Term := Proof.Events056.exact14570RawTerms
def group : MergeGroup := .operator 14756 14570
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14756) (leftOrdinal := 1)
    (rightResult := 14570) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6782⟩⟩, ⟨.program ⟨214⟩, ⟨7834⟩⟩, ⟨.program ⟨214⟩, ⟨25008⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14761

namespace LeftMerge14769
def owner : Owner := ⟨.program ⟨214⟩, ⟨26618⟩⟩
def mergeEvent : Nat := 14769
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩] } }
def leftRaw : List Term := Proof.Events057.exact14763RawTerms
def rightRaw : List Term := Proof.Events056.exact14467RawTerms
def group : MergeGroup := .operator 14763 14467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14763) (leftOrdinal := 1)
    (rightResult := 14467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26616⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14769

namespace LeftMerge14771
def owner : Owner := ⟨.program ⟨214⟩, ⟨26618⟩⟩
def mergeEvent : Nat := 14771
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23796⟩⟩] } }
def rhsRaw : List Term := Proof.Events056.exact14464RawTerms
def group : MergeGroup := .relation 14770
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14770) (rhsResult := 14464)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26616⟩⟩) ⟨23796⟩ 14464) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23796⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23796⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14771

namespace LeftMerge14772
def owner : Owner := ⟨.program ⟨214⟩, ⟨26618⟩⟩
def mergeEvent : Nat := 14772
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩] } }
def leftRaw : List Term := Proof.Events057.exact14763RawTerms
def rightRaw : List Term := Proof.Events056.exact14467RawTerms
def group : MergeGroup := .operator 14763 14467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14763) (leftOrdinal := 0)
    (rightResult := 14467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26616⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14772

namespace LeftMerge14786
def owner : Owner := ⟨.program ⟨214⟩, ⟨20555⟩⟩
def mergeEvent : Nat := 14786
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩] } }
def leftRaw : List Term := Proof.Events025.exact6561RawTerms
def rightRaw : List Term := Proof.Events057.exact14780RawTerms
def group : MergeGroup := .operator 6561 14780
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6561) (leftOrdinal := 0)
    (rightResult := 14780) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨20552⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14786

namespace LeftMerge14907
def owner : Owner := ⟨.program ⟨214⟩, ⟨15011⟩⟩
def mergeEvent : Nat := 14907
def frameStart : Nat := 14841
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events058.exact14903RawTerms
def rightRaw : List Term := Proof.Events058.exact14901RawTerms
def group : MergeGroup := .operator 14903 14901
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14903) (leftOrdinal := 0)
    (rightResult := 14901) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14907

namespace LeftMerge14919
def owner : Owner := ⟨.program ⟨214⟩, ⟨26617⟩⟩
def mergeEvent : Nat := 14919
def frameStart : Nat := 14841
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩] } }
def leftRaw : List Term := Proof.Events058.exact14915RawTerms
def rightRaw : List Term := Proof.Events058.exact14892RawTerms
def group : MergeGroup := .operator 14915 14892
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14915) (leftOrdinal := 1)
    (rightResult := 14892) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26616⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14919

namespace LeftMerge14921
def owner : Owner := ⟨.program ⟨214⟩, ⟨26617⟩⟩
def mergeEvent : Nat := 14921
def frameStart : Nat := 14841
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23796⟩⟩] } }
def rhsRaw : List Term := Proof.Events058.exact14889RawTerms
def group : MergeGroup := .relation 14920
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14920) (rhsResult := 14889)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨26616⟩⟩) ⟨23796⟩ 14889) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23796⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23796⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14921

namespace LeftMerge14922
def owner : Owner := ⟨.program ⟨214⟩, ⟨26617⟩⟩
def mergeEvent : Nat := 14922
def frameStart : Nat := 14841
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩] } }
def leftRaw : List Term := Proof.Events058.exact14915RawTerms
def rightRaw : List Term := Proof.Events058.exact14892RawTerms
def group : MergeGroup := .operator 14915 14892
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14915) (leftOrdinal := 0)
    (rightResult := 14892) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨26616⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14922

namespace LeftMerge14930
def owner : Owner := ⟨.program ⟨214⟩, ⟨15328⟩⟩
def mergeEvent : Nat := 14930
def frameStart : Nat := 14841
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events058.exact14903RawTerms
def rightRaw : List Term := Proof.Events058.exact14926RawTerms
def group : MergeGroup := .operator 14903 14926
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14903) (leftOrdinal := 0)
    (rightResult := 14926) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14930

namespace LeftMerge14947
def owner : Owner := ⟨.program ⟨214⟩, ⟨20555⟩⟩
def mergeEvent : Nat := 14947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23796⟩⟩] } }
def rhsRaw : List Term := Proof.Events058.exact14944RawTerms
def group : MergeGroup := .relation 14946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14946) (rhsResult := 14944)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 14945 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩) (none) 14944) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23796⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23796⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14947

namespace LeftMerge14948
def owner : Owner := ⟨.program ⟨214⟩, ⟨20555⟩⟩
def mergeEvent : Nat := 14948
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩] } }
def rhsRaw : List Term := Proof.Events058.exact14944RawTerms
def group : MergeGroup := .relation 14946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14946) (rhsResult := 14944)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 14945 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩) (none) 14944) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6691⟩⟩, ⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14948

namespace LeftMerge14949
def owner : Owner := ⟨.program ⟨214⟩, ⟨20555⟩⟩
def mergeEvent : Nat := 14949
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events058.exact14944RawTerms
def group : MergeGroup := .relation 14946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14946) (rhsResult := 14944)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 14945 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩) (none) 14944) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15326⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨15326⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14949

namespace LeftMerge14950
def owner : Owner := ⟨.program ⟨214⟩, ⟨20555⟩⟩
def mergeEvent : Nat := 14950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }
def rhsRaw : List Term := Proof.Events058.exact14944RawTerms
def group : MergeGroup := .relation 14946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 14946) (rhsResult := 14944)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 14945 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩) (none) 14944) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6711⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩], [⟨.program ⟨214⟩, ⟨6711⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge14950

namespace LeftMerge14955
def owner : Owner := ⟨.program ⟨214⟩, ⟨26619⟩⟩
def mergeEvent : Nat := 14955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23796⟩⟩] } }
def leftRaw : List Term := Proof.Events058.exact14951RawTerms
def rightRaw : List Term := Proof.Events057.exact14773RawTerms
def group : MergeGroup := .operator 14951 14773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 14951) (leftOrdinal := 2)
    (rightResult := 14773) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23796⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23796⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5519⟩⟩, ⟨.program ⟨214⟩, ⟨14969⟩⟩], [⟨.program ⟨214⟩, ⟨23796⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge14955

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
