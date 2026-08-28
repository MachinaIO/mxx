import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge299092
def owner : Owner := ⟨.program ⟨257⟩, ⟨64330⟩⟩
def mergeEvent : Nat := 299092
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63869⟩⟩] } }
def rhsRaw : List Term := Proof.Events1168.exact299017RawTerms
def group : MergeGroup := .relation 299091
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299091) (rhsResult := 299017)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64329⟩⟩) ⟨63869⟩ 299017) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63869⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], [⟨.program ⟨257⟩, ⟨63869⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299092

namespace LeftMerge299093
def owner : Owner := ⟨.program ⟨257⟩, ⟨64330⟩⟩
def mergeEvent : Nat := 299093
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩] } }
def leftRaw : List Term := Proof.Events1168.exact299084RawTerms
def rightRaw : List Term := Proof.Events1168.exact299020RawTerms
def group : MergeGroup := .operator 299084 299020
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299084) (leftOrdinal := 0)
    (rightResult := 299020) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299093

namespace LeftMerge299107
def owner : Owner := ⟨.program ⟨257⟩, ⟨63272⟩⟩
def mergeEvent : Nat := 299107
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1168.exact299101RawTerms
def group : MergeGroup := .operator 295195 299101
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 299101) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63269⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299107

namespace LeftMerge299162
def owner : Owner := ⟨.program ⟨257⟩, ⟨62196⟩⟩
def mergeEvent : Nat := 299162
def frameStart : Nat := 299144
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1168.exact299158RawTerms
def rightRaw : List Term := Proof.Events1168.exact299155RawTerms
def group : MergeGroup := .operator 299158 299155
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299158) (leftOrdinal := 0)
    (rightResult := 299155) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25370⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299162

namespace LeftMerge299192
def owner : Owner := ⟨.program ⟨257⟩, ⟨64168⟩⟩
def mergeEvent : Nat := 299192
def frameStart : Nat := 299144
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1168.exact299188RawTerms
def rightRaw : List Term := Proof.Events1168.exact299186RawTerms
def group : MergeGroup := .operator 299188 299186
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299188) (leftOrdinal := 0)
    (rightResult := 299186) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299192

namespace LeftMerge299215
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 299215
def frameStart : Nat := 299144
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events1168.exact299211RawTerms
def rightRaw : List Term := Proof.Events1168.exact299208RawTerms
def group : MergeGroup := .operator 299211 299208
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299211) (leftOrdinal := 0)
    (rightResult := 299208) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299215

namespace LeftMerge299224
def owner : Owner := ⟨.program ⟨257⟩, ⟨64332⟩⟩
def mergeEvent : Nat := 299224
def frameStart : Nat := 299144
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩] } }
def leftRaw : List Term := Proof.Events1168.exact299220RawTerms
def rightRaw : List Term := Proof.Events1168.exact299177RawTerms
def group : MergeGroup := .operator 299220 299177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299220) (leftOrdinal := 0)
    (rightResult := 299177) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64329⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299224

namespace LeftMerge299225
def owner : Owner := ⟨.program ⟨257⟩, ⟨64332⟩⟩
def mergeEvent : Nat := 299225
def frameStart : Nat := 299144
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩] } }
def leftRaw : List Term := Proof.Events1168.exact299220RawTerms
def rightRaw : List Term := Proof.Events1168.exact299177RawTerms
def group : MergeGroup := .operator 299220 299177
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299220) (leftOrdinal := 1)
    (rightResult := 299177) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299225

namespace LeftMerge299227
def owner : Owner := ⟨.program ⟨257⟩, ⟨64332⟩⟩
def mergeEvent : Nat := 299227
def frameStart : Nat := 299144
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63869⟩⟩] } }
def rhsRaw : List Term := Proof.Events1168.exact299174RawTerms
def group : MergeGroup := .relation 299226
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299226) (rhsResult := 299174)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64329⟩⟩) ⟨63869⟩ 299174) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63869⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], [⟨.program ⟨257⟩, ⟨63869⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299227

namespace LeftMerge299235
def owner : Owner := ⟨.program ⟨257⟩, ⟨62730⟩⟩
def mergeEvent : Nat := 299235
def frameStart : Nat := 299144
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1168.exact299188RawTerms
def rightRaw : List Term := Proof.Events1168.exact299231RawTerms
def group : MergeGroup := .operator 299188 299231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299188) (leftOrdinal := 0)
    (rightResult := 299231) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62728⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299235

namespace LeftMerge299252
def owner : Owner := ⟨.program ⟨257⟩, ⟨63272⟩⟩
def mergeEvent : Nat := 299252
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events1168.exact299249RawTerms
def group : MergeGroup := .relation 299251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299251) (rhsResult := 299249)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 299250 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩]⟩) (none) 299249) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299252

namespace LeftMerge299253
def owner : Owner := ⟨.program ⟨257⟩, ⟨63272⟩⟩
def mergeEvent : Nat := 299253
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩] } }
def rhsRaw : List Term := Proof.Events1168.exact299249RawTerms
def group : MergeGroup := .relation 299251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299251) (rhsResult := 299249)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 299250 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩]⟩) (none) 299249) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299253

namespace LeftMerge299254
def owner : Owner := ⟨.program ⟨257⟩, ⟨63272⟩⟩
def mergeEvent : Nat := 299254
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63869⟩⟩] } }
def rhsRaw : List Term := Proof.Events1168.exact299249RawTerms
def group : MergeGroup := .relation 299251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299251) (rhsResult := 299249)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 299250 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩]⟩) (none) 299249) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63869⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], [⟨.program ⟨257⟩, ⟨63869⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299254

namespace LeftMerge299255
def owner : Owner := ⟨.program ⟨257⟩, ⟨63272⟩⟩
def mergeEvent : Nat := 299255
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1168.exact299249RawTerms
def group : MergeGroup := .relation 299251
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 299251) (rhsResult := 299249)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 299250 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63269⟩⟩]⟩) (none) 299249) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62728⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨62728⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299255

namespace LeftMerge299260
def owner : Owner := ⟨.program ⟨257⟩, ⟨64331⟩⟩
def mergeEvent : Nat := 299260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63869⟩⟩] } }
def leftRaw : List Term := Proof.Events1168.exact299256RawTerms
def rightRaw : List Term := Proof.Events1168.exact299094RawTerms
def group : MergeGroup := .operator 299256 299094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299256) (leftOrdinal := 2)
    (rightResult := 299094) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63869⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63869⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨25370⟩⟩, ⟨.program ⟨257⟩, ⟨62195⟩⟩], [⟨.program ⟨257⟩, ⟨63869⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge299260

namespace LeftMerge299261
def owner : Owner := ⟨.program ⟨257⟩, ⟨64331⟩⟩
def mergeEvent : Nat := 299261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩] } }
def leftRaw : List Term := Proof.Events1168.exact299256RawTerms
def rightRaw : List Term := Proof.Events1168.exact299094RawTerms
def group : MergeGroup := .operator 299256 299094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 299256) (leftOrdinal := 1)
    (rightResult := 299094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64329⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge299261

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
