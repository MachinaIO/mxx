import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge97024
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def mergeEvent : Nat := 97024
def frameStart : Nat := 96941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }
def leftRaw : List Term := Proof.Events378.exact97020RawTerms
def rightRaw : List Term := Proof.Events378.exact97017RawTerms
def group : MergeGroup := .operator 97020 97017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97020) (leftOrdinal := 0)
    (rightResult := 97017) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9580⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97024

namespace LeftMerge97033
def owner : Owner := ⟨.program ⟨257⟩, ⟨52577⟩⟩
def mergeEvent : Nat := 97033
def frameStart : Nat := 96941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97029RawTerms
def rightRaw : List Term := Proof.Events378.exact96986RawTerms
def group : MergeGroup := .operator 97029 96986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97029) (leftOrdinal := 0)
    (rightResult := 96986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97033

namespace LeftMerge97034
def owner : Owner := ⟨.program ⟨257⟩, ⟨52577⟩⟩
def mergeEvent : Nat := 97034
def frameStart : Nat := 96941
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97029RawTerms
def rightRaw : List Term := Proof.Events378.exact96986RawTerms
def group : MergeGroup := .operator 97029 96986
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97029) (leftOrdinal := 1)
    (rightResult := 96986) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52574⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97034

namespace LeftMerge97036
def owner : Owner := ⟨.program ⟨257⟩, ⟨52577⟩⟩
def mergeEvent : Nat := 97036
def frameStart : Nat := 96941
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52039⟩⟩] } }
def rhsRaw : List Term := Proof.Events378.exact96983RawTerms
def group : MergeGroup := .relation 97035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97035) (rhsResult := 96983)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52574⟩⟩) ⟨52039⟩ 96983) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52039⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97036

namespace LeftMerge97044
def owner : Owner := ⟨.program ⟨257⟩, ⟨50930⟩⟩
def mergeEvent : Nat := 97044
def frameStart : Nat := 96941
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events378.exact96997RawTerms
def rightRaw : List Term := Proof.Events379.exact97040RawTerms
def group : MergeGroup := .operator 96997 97040
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 96997) (leftOrdinal := 0)
    (rightResult := 97040) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97044

namespace LeftMerge97061
def owner : Owner := ⟨.program ⟨257⟩, ⟨51502⟩⟩
def mergeEvent : Nat := 97061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events379.exact97058RawTerms
def group : MergeGroup := .relation 97060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97060) (rhsResult := 97058)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩) (none) 97058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97061

namespace LeftMerge97062
def owner : Owner := ⟨.program ⟨257⟩, ⟨51502⟩⟩
def mergeEvent : Nat := 97062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩] } }
def rhsRaw : List Term := Proof.Events379.exact97058RawTerms
def group : MergeGroup := .relation 97060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97060) (rhsResult := 97058)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩) (none) 97058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97062

namespace LeftMerge97063
def owner : Owner := ⟨.program ⟨257⟩, ⟨51502⟩⟩
def mergeEvent : Nat := 97063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52039⟩⟩] } }
def rhsRaw : List Term := Proof.Events379.exact97058RawTerms
def group : MergeGroup := .relation 97060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97060) (rhsResult := 97058)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩) (none) 97058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52039⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97063

namespace LeftMerge97064
def owner : Owner := ⟨.program ⟨257⟩, ⟨51502⟩⟩
def mergeEvent : Nat := 97064
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events379.exact97058RawTerms
def group : MergeGroup := .relation 97060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97060) (rhsResult := 97058)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 97059 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51499⟩⟩]⟩) (none) 97058) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97064

namespace LeftMerge97069
def owner : Owner := ⟨.program ⟨257⟩, ⟨52576⟩⟩
def mergeEvent : Nat := 97069
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52039⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97065RawTerms
def rightRaw : List Term := Proof.Events378.exact96879RawTerms
def group : MergeGroup := .operator 97065 96879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97065) (leftOrdinal := 2)
    (rightResult := 96879) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52039⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52039⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨24590⟩⟩, ⟨.program ⟨257⟩, ⟨50680⟩⟩], [⟨.program ⟨257⟩, ⟨52039⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97069

namespace LeftMerge97070
def owner : Owner := ⟨.program ⟨257⟩, ⟨52576⟩⟩
def mergeEvent : Nat := 97070
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97065RawTerms
def rightRaw : List Term := Proof.Events378.exact96879RawTerms
def group : MergeGroup := .operator 97065 96879
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97065) (leftOrdinal := 1)
    (rightResult := 96879) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97070

namespace LeftMerge97078
def owner : Owner := ⟨.program ⟨257⟩, ⟨53109⟩⟩
def mergeEvent : Nat := 97078
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97072RawTerms
def rightRaw : List Term := Proof.Events378.exact96795RawTerms
def group : MergeGroup := .operator 97072 96795
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97072) (leftOrdinal := 0)
    (rightResult := 96795) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53107⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97078

namespace LeftMerge97079
def owner : Owner := ⟨.program ⟨257⟩, ⟨53109⟩⟩
def mergeEvent : Nat := 97079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97072RawTerms
def rightRaw : List Term := Proof.Events378.exact96795RawTerms
def group : MergeGroup := .operator 97072 96795
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97072) (leftOrdinal := 1)
    (rightResult := 96795) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53107⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97079

namespace LeftMerge97081
def owner : Owner := ⟨.program ⟨257⟩, ⟨53109⟩⟩
def mergeEvent : Nat := 97081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52206⟩⟩] } }
def rhsRaw : List Term := Proof.Events378.exact96792RawTerms
def group : MergeGroup := .relation 97080
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 97080) (rhsResult := 96792)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53107⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53107⟩⟩) ⟨52206⟩ 96792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52206⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨52206⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge97081

namespace LeftMerge97095
def owner : Owner := ⟨.program ⟨257⟩, ⟨51859⟩⟩
def mergeEvent : Nat := 97095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51856⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events379.exact97089RawTerms
def group : MergeGroup := .operator 90620 97089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 97089) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51856⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51856⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97095

namespace LeftMerge97216
def owner : Owner := ⟨.program ⟨257⟩, ⟨52388⟩⟩
def mergeEvent : Nat := 97216
def frameStart : Nat := 97150
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events379.exact97212RawTerms
def rightRaw : List Term := Proof.Events379.exact97210RawTerms
def group : MergeGroup := .operator 97212 97210
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 97212) (leftOrdinal := 0)
    (rightResult := 97210) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50928⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50928⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge97216

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
