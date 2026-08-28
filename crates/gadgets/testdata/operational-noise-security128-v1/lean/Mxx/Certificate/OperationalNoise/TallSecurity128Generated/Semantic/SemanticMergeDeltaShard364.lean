import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge62000
def owner : Owner := ⟨.program ⟨257⟩, ⟨47059⟩⟩
def mergeEvent : Nat := 62000
def frameStart : Nat := 61907
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact61995RawTerms
def rightRaw : List Term := Proof.Events242.exact61952RawTerms
def group : MergeGroup := .operator 61995 61952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61995) (leftOrdinal := 1)
    (rightResult := 61952) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47056⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62000

namespace LeftMerge62002
def owner : Owner := ⟨.program ⟨257⟩, ⟨47059⟩⟩
def mergeEvent : Nat := 62002
def frameStart : Nat := 61907
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46511⟩⟩] } }
def rhsRaw : List Term := Proof.Events241.exact61949RawTerms
def group : MergeGroup := .relation 62001
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62001) (rhsResult := 61949)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47056⟩⟩) ⟨46511⟩ 61949) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46511⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨46511⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62002

namespace LeftMerge62010
def owner : Owner := ⟨.program ⟨257⟩, ⟨45526⟩⟩
def mergeEvent : Nat := 62010
def frameStart : Nat := 61907
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact61963RawTerms
def rightRaw : List Term := Proof.Events242.exact62006RawTerms
def group : MergeGroup := .operator 61963 62006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61963) (leftOrdinal := 0)
    (rightResult := 62006) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62010

namespace LeftMerge62027
def owner : Owner := ⟨.program ⟨257⟩, ⟨45982⟩⟩
def mergeEvent : Nat := 62027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events242.exact62024RawTerms
def group : MergeGroup := .relation 62026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62026) (rhsResult := 62024)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩]⟩) (none) 62024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62027

namespace LeftMerge62028
def owner : Owner := ⟨.program ⟨257⟩, ⟨45982⟩⟩
def mergeEvent : Nat := 62028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩] } }
def rhsRaw : List Term := Proof.Events242.exact62024RawTerms
def group : MergeGroup := .relation 62026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62026) (rhsResult := 62024)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩]⟩) (none) 62024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62028

namespace LeftMerge62029
def owner : Owner := ⟨.program ⟨257⟩, ⟨45982⟩⟩
def mergeEvent : Nat := 62029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46511⟩⟩] } }
def rhsRaw : List Term := Proof.Events242.exact62024RawTerms
def group : MergeGroup := .relation 62026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62026) (rhsResult := 62024)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩]⟩) (none) 62024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46511⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨46511⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62029

namespace LeftMerge62030
def owner : Owner := ⟨.program ⟨257⟩, ⟨45982⟩⟩
def mergeEvent : Nat := 62030
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events242.exact62024RawTerms
def group : MergeGroup := .relation 62026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62026) (rhsResult := 62024)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 62025 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45979⟩⟩]⟩) (none) 62024) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62030

namespace LeftMerge62035
def owner : Owner := ⟨.program ⟨257⟩, ⟨47058⟩⟩
def mergeEvent : Nat := 62035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46511⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62031RawTerms
def rightRaw : List Term := Proof.Events241.exact61845RawTerms
def group : MergeGroup := .operator 62031 61845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62031) (leftOrdinal := 2)
    (rightResult := 61845) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46511⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46511⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨14886⟩⟩, ⟨.program ⟨257⟩, ⟨45322⟩⟩], [⟨.program ⟨257⟩, ⟨46511⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62035

namespace LeftMerge62036
def owner : Owner := ⟨.program ⟨257⟩, ⟨47058⟩⟩
def mergeEvent : Nat := 62036
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62031RawTerms
def rightRaw : List Term := Proof.Events241.exact61845RawTerms
def group : MergeGroup := .operator 62031 61845
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62031) (leftOrdinal := 1)
    (rightResult := 61845) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47056⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62036

namespace LeftMerge62044
def owner : Owner := ⟨.program ⟨257⟩, ⟨47526⟩⟩
def mergeEvent : Nat := 62044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62038RawTerms
def rightRaw : List Term := Proof.Events241.exact61761RawTerms
def group : MergeGroup := .operator 62038 61761
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62038) (leftOrdinal := 0)
    (rightResult := 61761) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47524⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62044

namespace LeftMerge62045
def owner : Owner := ⟨.program ⟨257⟩, ⟨47526⟩⟩
def mergeEvent : Nat := 62045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62038RawTerms
def rightRaw : List Term := Proof.Events241.exact61761RawTerms
def group : MergeGroup := .operator 62038 61761
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62038) (leftOrdinal := 1)
    (rightResult := 61761) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47524⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62045

namespace LeftMerge62047
def owner : Owner := ⟨.program ⟨257⟩, ⟨47526⟩⟩
def mergeEvent : Nat := 62047
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46684⟩⟩] } }
def rhsRaw : List Term := Proof.Events241.exact61758RawTerms
def group : MergeGroup := .relation 62046
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 62046) (rhsResult := 61758)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47524⟩⟩) ⟨46684⟩ 61758) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46684⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨46684⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62047

namespace LeftMerge62061
def owner : Owner := ⟨.program ⟨257⟩, ⟨46359⟩⟩
def mergeEvent : Nat := 62061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events242.exact62055RawTerms
def group : MergeGroup := .operator 61370 62055
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 62055) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46356⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46356⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62061

namespace LeftMerge62182
def owner : Owner := ⟨.program ⟨257⟩, ⟨46856⟩⟩
def mergeEvent : Nat := 62182
def frameStart : Nat := 62116
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62178RawTerms
def rightRaw : List Term := Proof.Events242.exact62176RawTerms
def group : MergeGroup := .operator 62178 62176
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62178) (leftOrdinal := 0)
    (rightResult := 62176) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62182

namespace LeftMerge62194
def owner : Owner := ⟨.program ⟨257⟩, ⟨47525⟩⟩
def mergeEvent : Nat := 62194
def frameStart : Nat := 62116
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62190RawTerms
def rightRaw : List Term := Proof.Events242.exact62167RawTerms
def group : MergeGroup := .operator 62190 62167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62190) (leftOrdinal := 0)
    (rightResult := 62167) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47524⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge62194

namespace LeftMerge62195
def owner : Owner := ⟨.program ⟨257⟩, ⟨47525⟩⟩
def mergeEvent : Nat := 62195
def frameStart : Nat := 62116
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62190RawTerms
def rightRaw : List Term := Proof.Events242.exact62167RawTerms
def group : MergeGroup := .operator 62190 62167
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62190) (leftOrdinal := 1)
    (rightResult := 62167) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47524⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47524⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge62195

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
