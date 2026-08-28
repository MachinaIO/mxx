import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge148848
def owner : Owner := ⟨.program ⟨257⟩, ⟨71027⟩⟩
def mergeEvent : Nat := 148848
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16287RawTerms
def group : MergeGroup := .relation 148847
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 148847) (rhsResult := 16287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7117⟩⟩) ⟨7024⟩ 16287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨31968⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148848

namespace LeftMerge148849
def owner : Owner := ⟨.program ⟨257⟩, ⟨71027⟩⟩
def mergeEvent : Nat := 148849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148795RawTerms
def rightRaw : List Term := Proof.Events063.exact16294RawTerms
def group : MergeGroup := .operator 148795 16294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148795) (leftOrdinal := 11)
    (rightResult := 16294) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7117⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148849

namespace LeftMerge148851
def owner : Owner := ⟨.program ⟨257⟩, ⟨71027⟩⟩
def mergeEvent : Nat := 148851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16287RawTerms
def group : MergeGroup := .relation 148850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 148850) (rhsResult := 16287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7117⟩⟩) ⟨7024⟩ 16287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨21948⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148851

namespace LeftMerge148852
def owner : Owner := ⟨.program ⟨257⟩, ⟨71027⟩⟩
def mergeEvent : Nat := 148852
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148795RawTerms
def rightRaw : List Term := Proof.Events063.exact16294RawTerms
def group : MergeGroup := .operator 148795 16294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148795) (leftOrdinal := 15)
    (rightResult := 16294) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7117⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148852

namespace LeftMerge148854
def owner : Owner := ⟨.program ⟨257⟩, ⟨71027⟩⟩
def mergeEvent : Nat := 148854
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16287RawTerms
def group : MergeGroup := .relation 148853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 148853) (rhsResult := 16287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7117⟩⟩) ⟨7024⟩ 16287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨18728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148854

namespace LeftMerge148855
def owner : Owner := ⟨.program ⟨257⟩, ⟨71027⟩⟩
def mergeEvent : Nat := 148855
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148795RawTerms
def rightRaw : List Term := Proof.Events063.exact16294RawTerms
def group : MergeGroup := .operator 148795 16294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148795) (leftOrdinal := 18)
    (rightResult := 16294) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7117⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148855

namespace LeftMerge148857
def owner : Owner := ⟨.program ⟨257⟩, ⟨71027⟩⟩
def mergeEvent : Nat := 148857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16287RawTerms
def group : MergeGroup := .relation 148856
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 148856) (rhsResult := 16287)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7117⟩⟩) ⟨7024⟩ 16287) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6751⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6751⟩⟩, ⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨15918⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148857

namespace LeftMerge148858
def owner : Owner := ⟨.program ⟨257⟩, ⟨71027⟩⟩
def mergeEvent : Nat := 148858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9507⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148795RawTerms
def rightRaw : List Term := Proof.Events063.exact16294RawTerms
def group : MergeGroup := .operator 148795 16294
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148795) (leftOrdinal := 0)
    (rightResult := 16294) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9507⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7117⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7250⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9507⟩⟩, ⟨.program ⟨257⟩, ⟨7117⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148858

namespace LeftMerge148886
def owner : Owner := ⟨.program ⟨257⟩, ⟨6980⟩⟩
def mergeEvent : Nat := 148886
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148882RawTerms
def rightRaw : List Term := Proof.Events000.exact2RawTerms
def group : MergeGroup := .operator 148882 2
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148882) (leftOrdinal := 0)
    (rightResult := 2) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148886

namespace LeftMerge148902
def owner : Owner := ⟨.program ⟨257⟩, ⟨8235⟩⟩
def mergeEvent : Nat := 148902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148898RawTerms
def rightRaw : List Term := Proof.Events063.exact16337RawTerms
def group : MergeGroup := .operator 148898 16337
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148898) (leftOrdinal := 0)
    (rightResult := 16337) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7253⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7253⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148902

namespace LeftMerge148955
def owner : Owner := ⟨.program ⟨257⟩, ⟨67404⟩⟩
def mergeEvent : Nat := 148955
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148913RawTerms
def rightRaw : List Term := Proof.Events029.exact7536RawTerms
def group : MergeGroup := .operator 148913 7536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148913) (leftOrdinal := 0)
    (rightResult := 7536) (rightOrdinal := 5) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨67399⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge148955

namespace LeftMerge148956
def owner : Owner := ⟨.program ⟨257⟩, ⟨67404⟩⟩
def mergeEvent : Nat := 148956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148913RawTerms
def rightRaw : List Term := Proof.Events029.exact7536RawTerms
def group : MergeGroup := .operator 148913 7536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148913) (leftOrdinal := 0)
    (rightResult := 7536) (rightOrdinal := 7) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨48320⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148956

namespace LeftMerge148957
def owner : Owner := ⟨.program ⟨257⟩, ⟨67404⟩⟩
def mergeEvent : Nat := 148957
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148913RawTerms
def rightRaw : List Term := Proof.Events029.exact7536RawTerms
def group : MergeGroup := .operator 148913 7536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148913) (leftOrdinal := 0)
    (rightResult := 7536) (rightOrdinal := 8) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨45640⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148957

namespace LeftMerge148958
def owner : Owner := ⟨.program ⟨257⟩, ⟨67404⟩⟩
def mergeEvent : Nat := 148958
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148913RawTerms
def rightRaw : List Term := Proof.Events029.exact7536RawTerms
def group : MergeGroup := .operator 148913 7536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148913) (leftOrdinal := 0)
    (rightResult := 7536) (rightOrdinal := 9) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨42963⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148958

namespace LeftMerge148959
def owner : Owner := ⟨.program ⟨257⟩, ⟨67404⟩⟩
def mergeEvent : Nat := 148959
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148913RawTerms
def rightRaw : List Term := Proof.Events029.exact7536RawTerms
def group : MergeGroup := .operator 148913 7536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148913) (leftOrdinal := 0)
    (rightResult := 7536) (rightOrdinal := 11) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨40283⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148959

namespace LeftMerge148960
def owner : Owner := ⟨.program ⟨257⟩, ⟨67404⟩⟩
def mergeEvent : Nat := 148960
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events581.exact148913RawTerms
def rightRaw : List Term := Proof.Events029.exact7536RawTerms
def group : MergeGroup := .operator 148913 7536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 148913) (leftOrdinal := 0)
    (rightResult := 7536) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge148960

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
