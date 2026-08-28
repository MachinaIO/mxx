import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge244089
def owner : Owner := ⟨.program ⟨257⟩, ⟨23418⟩⟩
def mergeEvent : Nat := 244089
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩] } }
def leftRaw : List Term := Proof.Events953.exact244083RawTerms
def rightRaw : List Term := Proof.Events953.exact244019RawTerms
def group : MergeGroup := .operator 244083 244019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 244083) (leftOrdinal := 1)
    (rightResult := 244019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23417⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge244089

namespace LeftMerge244091
def owner : Owner := ⟨.program ⟨257⟩, ⟨23418⟩⟩
def mergeEvent : Nat := 244091
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22917⟩⟩] } }
def rhsRaw : List Term := Proof.Events953.exact244016RawTerms
def group : MergeGroup := .relation 244090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 244090) (rhsResult := 244016)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23417⟩⟩) ⟨22917⟩ 244016) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22917⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], [⟨.program ⟨257⟩, ⟨22917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge244091

namespace LeftMerge244092
def owner : Owner := ⟨.program ⟨257⟩, ⟨23418⟩⟩
def mergeEvent : Nat := 244092
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩] } }
def leftRaw : List Term := Proof.Events953.exact244083RawTerms
def rightRaw : List Term := Proof.Events953.exact244019RawTerms
def group : MergeGroup := .operator 244083 244019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 244083) (leftOrdinal := 0)
    (rightResult := 244019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23417⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge244092

namespace LeftMerge244106
def owner : Owner := ⟨.program ⟨257⟩, ⟨22352⟩⟩
def mergeEvent : Nat := 244106
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events953.exact244100RawTerms
def group : MergeGroup := .operator 236870 244100
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 244100) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22349⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge244106

namespace LeftMerge244185
def owner : Owner := ⟨.program ⟨257⟩, ⟨21447⟩⟩
def mergeEvent : Nat := 244185
def frameStart : Nat := 244155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events953.exact244181RawTerms
def rightRaw : List Term := Proof.Events953.exact244178RawTerms
def group : MergeGroup := .operator 244181 244178
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 244181) (leftOrdinal := 0)
    (rightResult := 244178) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21071⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge244185

namespace LeftMerge244215
def owner : Owner := ⟨.program ⟨257⟩, ⟨23200⟩⟩
def mergeEvent : Nat := 244215
def frameStart : Nat := 244155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events953.exact244211RawTerms
def rightRaw : List Term := Proof.Events953.exact244209RawTerms
def group : MergeGroup := .operator 244211 244209
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 244211) (leftOrdinal := 0)
    (rightResult := 244209) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge244215

namespace LeftMerge244238
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 244238
def frameStart : Nat := 244155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events954.exact244234RawTerms
def rightRaw : List Term := Proof.Events954.exact244231RawTerms
def group : MergeGroup := .operator 244234 244231
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 244234) (leftOrdinal := 0)
    (rightResult := 244231) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge244238

namespace LeftMerge244247
def owner : Owner := ⟨.program ⟨257⟩, ⟨23420⟩⟩
def mergeEvent : Nat := 244247
def frameStart : Nat := 244155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩] } }
def leftRaw : List Term := Proof.Events954.exact244243RawTerms
def rightRaw : List Term := Proof.Events953.exact244200RawTerms
def group : MergeGroup := .operator 244243 244200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 244243) (leftOrdinal := 0)
    (rightResult := 244200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23417⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge244247

namespace LeftMerge244248
def owner : Owner := ⟨.program ⟨257⟩, ⟨23420⟩⟩
def mergeEvent : Nat := 244248
def frameStart : Nat := 244155
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩] } }
def leftRaw : List Term := Proof.Events954.exact244243RawTerms
def rightRaw : List Term := Proof.Events953.exact244200RawTerms
def group : MergeGroup := .operator 244243 244200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 244243) (leftOrdinal := 1)
    (rightResult := 244200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23417⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge244248

namespace LeftMerge244250
def owner : Owner := ⟨.program ⟨257⟩, ⟨23420⟩⟩
def mergeEvent : Nat := 244250
def frameStart : Nat := 244155
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22917⟩⟩] } }
def rhsRaw : List Term := Proof.Events953.exact244197RawTerms
def group : MergeGroup := .relation 244249
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 244249) (rhsResult := 244197)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23417⟩⟩) ⟨22917⟩ 244197) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22917⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], [⟨.program ⟨257⟩, ⟨22917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge244250

namespace LeftMerge244258
def owner : Owner := ⟨.program ⟨257⟩, ⟨21794⟩⟩
def mergeEvent : Nat := 244258
def frameStart : Nat := 244155
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events953.exact244211RawTerms
def rightRaw : List Term := Proof.Events954.exact244254RawTerms
def group : MergeGroup := .operator 244211 244254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 244211) (leftOrdinal := 0)
    (rightResult := 244254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21792⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge244258

namespace LeftMerge244275
def owner : Owner := ⟨.program ⟨257⟩, ⟨22352⟩⟩
def mergeEvent : Nat := 244275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events954.exact244272RawTerms
def group : MergeGroup := .relation 244274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 244274) (rhsResult := 244272)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 244273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩) (none) 244272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge244275

namespace LeftMerge244276
def owner : Owner := ⟨.program ⟨257⟩, ⟨22352⟩⟩
def mergeEvent : Nat := 244276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩] } }
def rhsRaw : List Term := Proof.Events954.exact244272RawTerms
def group : MergeGroup := .relation 244274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 244274) (rhsResult := 244272)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 244273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩) (none) 244272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23417⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge244276

namespace LeftMerge244277
def owner : Owner := ⟨.program ⟨257⟩, ⟨22352⟩⟩
def mergeEvent : Nat := 244277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22917⟩⟩] } }
def rhsRaw : List Term := Proof.Events954.exact244272RawTerms
def group : MergeGroup := .relation 244274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 244274) (rhsResult := 244272)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 244273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩) (none) 244272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22917⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], [⟨.program ⟨257⟩, ⟨22917⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge244277

namespace LeftMerge244278
def owner : Owner := ⟨.program ⟨257⟩, ⟨22352⟩⟩
def mergeEvent : Nat := 244278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events954.exact244272RawTerms
def group : MergeGroup := .relation 244274
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 244274) (rhsResult := 244272)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 244273 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩) (none) 244272) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21792⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21792⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge244278

namespace LeftMerge244283
def owner : Owner := ⟨.program ⟨257⟩, ⟨23419⟩⟩
def mergeEvent : Nat := 244283
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22917⟩⟩] } }
def leftRaw : List Term := Proof.Events954.exact244279RawTerms
def rightRaw : List Term := Proof.Events953.exact244093RawTerms
def group : MergeGroup := .operator 244279 244093
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 244279) (leftOrdinal := 2)
    (rightResult := 244093) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22917⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22917⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨21071⟩⟩, ⟨.program ⟨257⟩, ⟨21446⟩⟩], [⟨.program ⟨257⟩, ⟨22917⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge244283

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
