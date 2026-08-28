import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge104886
def owner : Owner := ⟨.program ⟨257⟩, ⟨71413⟩⟩
def mergeEvent : Nat := 104886
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104832RawTerms
def rightRaw : List Term := Proof.Events063.exact16184RawTerms
def group : MergeGroup := .operator 104832 16184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104832) (leftOrdinal := 11)
    (rightResult := 16184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104886

namespace LeftMerge104888
def owner : Owner := ⟨.program ⟨257⟩, ⟨71413⟩⟩
def mergeEvent : Nat := 104888
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16177RawTerms
def group : MergeGroup := .relation 104887
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 104887) (rhsResult := 16177)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9501⟩⟩) ⟨7245⟩ 16177) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨22176⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104888

namespace LeftMerge104889
def owner : Owner := ⟨.program ⟨257⟩, ⟨71413⟩⟩
def mergeEvent : Nat := 104889
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104832RawTerms
def rightRaw : List Term := Proof.Events063.exact16184RawTerms
def group : MergeGroup := .operator 104832 16184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104832) (leftOrdinal := 15)
    (rightResult := 16184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104889

namespace LeftMerge104891
def owner : Owner := ⟨.program ⟨257⟩, ⟨71413⟩⟩
def mergeEvent : Nat := 104891
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16177RawTerms
def group : MergeGroup := .relation 104890
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 104890) (rhsResult := 16177)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9501⟩⟩) ⟨7245⟩ 16177) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨18956⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104891

namespace LeftMerge104892
def owner : Owner := ⟨.program ⟨257⟩, ⟨71413⟩⟩
def mergeEvent : Nat := 104892
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104832RawTerms
def rightRaw : List Term := Proof.Events063.exact16184RawTerms
def group : MergeGroup := .operator 104832 16184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104832) (leftOrdinal := 18)
    (rightResult := 16184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104892

namespace LeftMerge104894
def owner : Owner := ⟨.program ⟨257⟩, ⟨71413⟩⟩
def mergeEvent : Nat := 104894
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16177RawTerms
def group : MergeGroup := .relation 104893
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 104893) (rhsResult := 16177)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9501⟩⟩) ⟨7245⟩ 16177) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16110⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104894

namespace LeftMerge104895
def owner : Owner := ⟨.program ⟨257⟩, ⟨71413⟩⟩
def mergeEvent : Nat := 104895
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104832RawTerms
def rightRaw : List Term := Proof.Events063.exact16184RawTerms
def group : MergeGroup := .operator 104832 16184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104832) (leftOrdinal := 0)
    (rightResult := 16184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7244⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9501⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104895

namespace LeftMerge104900
def owner : Owner := ⟨.program ⟨257⟩, ⟨71414⟩⟩
def mergeEvent : Nat := 104900
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104896RawTerms
def rightRaw : List Term := Proof.Events353.exact90493RawTerms
def group : MergeGroup := .operator 104896 90493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104896) (leftOrdinal := 6)
    (rightResult := 90493) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨67566⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge104900

namespace LeftMerge104901
def owner : Owner := ⟨.program ⟨257⟩, ⟨71414⟩⟩
def mergeEvent : Nat := 104901
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104896RawTerms
def rightRaw : List Term := Proof.Events353.exact90493RawTerms
def group : MergeGroup := .operator 104896 90493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104896) (leftOrdinal := 8)
    (rightResult := 90493) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨48424⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104901

namespace LeftMerge104902
def owner : Owner := ⟨.program ⟨257⟩, ⟨71414⟩⟩
def mergeEvent : Nat := 104902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104896RawTerms
def rightRaw : List Term := Proof.Events353.exact90493RawTerms
def group : MergeGroup := .operator 104896 90493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104896) (leftOrdinal := 9)
    (rightResult := 90493) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨45744⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104902

namespace LeftMerge104903
def owner : Owner := ⟨.program ⟨257⟩, ⟨71414⟩⟩
def mergeEvent : Nat := 104903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104896RawTerms
def rightRaw : List Term := Proof.Events353.exact90493RawTerms
def group : MergeGroup := .operator 104896 90493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104896) (leftOrdinal := 10)
    (rightResult := 90493) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨43067⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104903

namespace LeftMerge104904
def owner : Owner := ⟨.program ⟨257⟩, ⟨71414⟩⟩
def mergeEvent : Nat := 104904
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104896RawTerms
def rightRaw : List Term := Proof.Events353.exact90493RawTerms
def group : MergeGroup := .operator 104896 90493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104896) (leftOrdinal := 12)
    (rightResult := 90493) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨40387⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104904

namespace LeftMerge104905
def owner : Owner := ⟨.program ⟨257⟩, ⟨71414⟩⟩
def mergeEvent : Nat := 104905
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104896RawTerms
def rightRaw : List Term := Proof.Events353.exact90493RawTerms
def group : MergeGroup := .operator 104896 90493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104896) (leftOrdinal := 13)
    (rightResult := 90493) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨37704⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104905

namespace LeftMerge104906
def owner : Owner := ⟨.program ⟨257⟩, ⟨71414⟩⟩
def mergeEvent : Nat := 104906
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104896RawTerms
def rightRaw : List Term := Proof.Events353.exact90493RawTerms
def group : MergeGroup := .operator 104896 90493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104896) (leftOrdinal := 14)
    (rightResult := 90493) (rightOrdinal := 32) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨35024⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104906

namespace LeftMerge104907
def owner : Owner := ⟨.program ⟨257⟩, ⟨71414⟩⟩
def mergeEvent : Nat := 104907
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104896RawTerms
def rightRaw : List Term := Proof.Events353.exact90493RawTerms
def group : MergeGroup := .operator 104896 90493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104896) (leftOrdinal := 16)
    (rightResult := 90493) (rightOrdinal := 34) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨29367⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104907

namespace LeftMerge104908
def owner : Owner := ⟨.program ⟨257⟩, ⟨71414⟩⟩
def mergeEvent : Nat := 104908
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }
def leftRaw : List Term := Proof.Events409.exact104896RawTerms
def rightRaw : List Term := Proof.Events353.exact90493RawTerms
def group : MergeGroup := .operator 104896 90493
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 104896) (leftOrdinal := 17)
    (rightResult := 90493) (rightOrdinal := 35) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7245⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨26687⟩⟩], [⟨.program ⟨257⟩, ⟨7245⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge104908

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
