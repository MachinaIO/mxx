import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge272857
def owner : Owner := ⟨.program ⟨257⟩, ⟨33369⟩⟩
def mergeEvent : Nat := 272857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩] } }
def leftRaw : List Term := Proof.Events1065.exact272851RawTerms
def rightRaw : List Term := Proof.Events1065.exact272787RawTerms
def group : MergeGroup := .operator 272851 272787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272851) (leftOrdinal := 1)
    (rightResult := 272787) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33368⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272857

namespace LeftMerge272859
def owner : Owner := ⟨.program ⟨257⟩, ⟨33369⟩⟩
def mergeEvent : Nat := 272859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32899⟩⟩] } }
def rhsRaw : List Term := Proof.Events1065.exact272784RawTerms
def group : MergeGroup := .relation 272858
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272858) (rhsResult := 272784)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33368⟩⟩) ⟨32899⟩ 272784) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32899⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], [⟨.program ⟨257⟩, ⟨32899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272859

namespace LeftMerge272860
def owner : Owner := ⟨.program ⟨257⟩, ⟨33369⟩⟩
def mergeEvent : Nat := 272860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩] } }
def leftRaw : List Term := Proof.Events1065.exact272851RawTerms
def rightRaw : List Term := Proof.Events1065.exact272787RawTerms
def group : MergeGroup := .operator 272851 272787
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272851) (leftOrdinal := 0)
    (rightResult := 272787) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33368⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272860

namespace LeftMerge272874
def owner : Owner := ⟨.program ⟨257⟩, ⟨32309⟩⟩
def mergeEvent : Nat := 272874
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1065.exact272868RawTerms
def group : MergeGroup := .operator 266120 272868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 272868) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272874

namespace LeftMerge272953
def owner : Owner := ⟨.program ⟨257⟩, ⟨31261⟩⟩
def mergeEvent : Nat := 272953
def frameStart : Nat := 272923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1066.exact272949RawTerms
def rightRaw : List Term := Proof.Events1066.exact272946RawTerms
def group : MergeGroup := .operator 272949 272946
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272949) (leftOrdinal := 0)
    (rightResult := 272946) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24190⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272953

namespace LeftMerge272983
def owner : Owner := ⟨.program ⟨257⟩, ⟨33196⟩⟩
def mergeEvent : Nat := 272983
def frameStart : Nat := 272923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1066.exact272979RawTerms
def rightRaw : List Term := Proof.Events1066.exact272977RawTerms
def group : MergeGroup := .operator 272979 272977
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272979) (leftOrdinal := 0)
    (rightResult := 272977) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272983

namespace LeftMerge273006
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def mergeEvent : Nat := 273006
def frameStart : Nat := 272923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }
def leftRaw : List Term := Proof.Events1066.exact273002RawTerms
def rightRaw : List Term := Proof.Events1066.exact272999RawTerms
def group : MergeGroup := .operator 273002 272999
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273002) (leftOrdinal := 0)
    (rightResult := 272999) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9577⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273006

namespace LeftMerge273015
def owner : Owner := ⟨.program ⟨257⟩, ⟨33371⟩⟩
def mergeEvent : Nat := 273015
def frameStart : Nat := 272923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩] } }
def leftRaw : List Term := Proof.Events1066.exact273011RawTerms
def rightRaw : List Term := Proof.Events1066.exact272968RawTerms
def group : MergeGroup := .operator 273011 272968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273011) (leftOrdinal := 0)
    (rightResult := 272968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33368⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273015

namespace LeftMerge273016
def owner : Owner := ⟨.program ⟨257⟩, ⟨33371⟩⟩
def mergeEvent : Nat := 273016
def frameStart : Nat := 272923
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩] } }
def leftRaw : List Term := Proof.Events1066.exact273011RawTerms
def rightRaw : List Term := Proof.Events1066.exact272968RawTerms
def group : MergeGroup := .operator 273011 272968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273011) (leftOrdinal := 1)
    (rightResult := 272968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33368⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273016

namespace LeftMerge273018
def owner : Owner := ⟨.program ⟨257⟩, ⟨33371⟩⟩
def mergeEvent : Nat := 273018
def frameStart : Nat := 272923
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32899⟩⟩] } }
def rhsRaw : List Term := Proof.Events1066.exact272965RawTerms
def group : MergeGroup := .relation 273017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273017) (rhsResult := 272965)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33368⟩⟩) ⟨32899⟩ 272965) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32899⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], [⟨.program ⟨257⟩, ⟨32899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273018

namespace LeftMerge273026
def owner : Owner := ⟨.program ⟨257⟩, ⟨31764⟩⟩
def mergeEvent : Nat := 273026
def frameStart : Nat := 272923
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1066.exact272979RawTerms
def rightRaw : List Term := Proof.Events1066.exact273022RawTerms
def group : MergeGroup := .operator 272979 273022
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272979) (leftOrdinal := 0)
    (rightResult := 273022) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31762⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273026

namespace LeftMerge273043
def owner : Owner := ⟨.program ⟨257⟩, ⟨32309⟩⟩
def mergeEvent : Nat := 273043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }
def rhsRaw : List Term := Proof.Events1066.exact273040RawTerms
def group : MergeGroup := .relation 273042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273042) (rhsResult := 273040)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 273041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩]⟩) (none) 273040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273043

namespace LeftMerge273044
def owner : Owner := ⟨.program ⟨257⟩, ⟨32309⟩⟩
def mergeEvent : Nat := 273044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩] } }
def rhsRaw : List Term := Proof.Events1066.exact273040RawTerms
def group : MergeGroup := .relation 273042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273042) (rhsResult := 273040)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 273041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩]⟩) (none) 273040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7287⟩⟩, ⟨.program ⟨257⟩, ⟨9577⟩⟩, ⟨.program ⟨257⟩, ⟨33368⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273044

namespace LeftMerge273045
def owner : Owner := ⟨.program ⟨257⟩, ⟨32309⟩⟩
def mergeEvent : Nat := 273045
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32899⟩⟩] } }
def rhsRaw : List Term := Proof.Events1066.exact273040RawTerms
def group : MergeGroup := .relation 273042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273042) (rhsResult := 273040)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 273041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩]⟩) (none) 273040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32899⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], [⟨.program ⟨257⟩, ⟨32899⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge273045

namespace LeftMerge273046
def owner : Owner := ⟨.program ⟨257⟩, ⟨32309⟩⟩
def mergeEvent : Nat := 273046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1066.exact273040RawTerms
def group : MergeGroup := .relation 273042
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 273042) (rhsResult := 273040)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 273041 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32306⟩⟩]⟩) (none) 273040) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31762⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨31762⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273046

namespace LeftMerge273051
def owner : Owner := ⟨.program ⟨257⟩, ⟨33370⟩⟩
def mergeEvent : Nat := 273051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32899⟩⟩] } }
def leftRaw : List Term := Proof.Events1066.exact273047RawTerms
def rightRaw : List Term := Proof.Events1065.exact272861RawTerms
def group : MergeGroup := .operator 273047 272861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 273047) (leftOrdinal := 2)
    (rightResult := 272861) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32899⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨32899⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24190⟩⟩, ⟨.program ⟨257⟩, ⟨31260⟩⟩], [⟨.program ⟨257⟩, ⟨32899⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge273051

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
