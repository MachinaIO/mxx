import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge170966
def owner : Owner := ⟨.program ⟨257⟩, ⟨23484⟩⟩
def mergeEvent : Nat := 170966
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22953⟩⟩] } }
def rhsRaw : List Term := Proof.Events667.exact170891RawTerms
def group : MergeGroup := .relation 170965
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 170965) (rhsResult := 170891)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23483⟩⟩) ⟨22953⟩ 170891) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22953⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], [⟨.program ⟨257⟩, ⟨22953⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge170966

namespace LeftMerge170967
def owner : Owner := ⟨.program ⟨257⟩, ⟨23484⟩⟩
def mergeEvent : Nat := 170967
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩] } }
def leftRaw : List Term := Proof.Events667.exact170958RawTerms
def rightRaw : List Term := Proof.Events667.exact170894RawTerms
def group : MergeGroup := .operator 170958 170894
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 170958) (leftOrdinal := 0)
    (rightResult := 170894) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23483⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170967

namespace LeftMerge170981
def owner : Owner := ⟨.program ⟨257⟩, ⟨22412⟩⟩
def mergeEvent : Nat := 170981
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩] } }
def leftRaw : List Term := Proof.Events639.exact163745RawTerms
def rightRaw : List Term := Proof.Events667.exact170975RawTerms
def group : MergeGroup := .operator 163745 170975
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163745) (leftOrdinal := 0)
    (rightResult := 170975) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22409⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge170981

namespace LeftMerge171060
def owner : Owner := ⟨.program ⟨257⟩, ⟨21591⟩⟩
def mergeEvent : Nat := 171060
def frameStart : Nat := 171030
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events668.exact171056RawTerms
def rightRaw : List Term := Proof.Events668.exact171053RawTerms
def group : MergeGroup := .operator 171056 171053
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171056) (leftOrdinal := 0)
    (rightResult := 171053) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21161⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171060

namespace LeftMerge171090
def owner : Owner := ⟨.program ⟨257⟩, ⟨23224⟩⟩
def mergeEvent : Nat := 171090
def frameStart : Nat := 171030
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171086RawTerms
def rightRaw : List Term := Proof.Events668.exact171084RawTerms
def group : MergeGroup := .operator 171086 171084
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171086) (leftOrdinal := 0)
    (rightResult := 171084) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171090

namespace LeftMerge171113
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def mergeEvent : Nat := 171113
def frameStart : Nat := 171030
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171109RawTerms
def rightRaw : List Term := Proof.Events668.exact171106RawTerms
def group : MergeGroup := .operator 171109 171106
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171109) (leftOrdinal := 0)
    (rightResult := 171106) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9574⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171113

namespace LeftMerge171122
def owner : Owner := ⟨.program ⟨257⟩, ⟨23486⟩⟩
def mergeEvent : Nat := 171122
def frameStart : Nat := 171030
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171118RawTerms
def rightRaw : List Term := Proof.Events668.exact171075RawTerms
def group : MergeGroup := .operator 171118 171075
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171118) (leftOrdinal := 0)
    (rightResult := 171075) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23483⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171122

namespace LeftMerge171123
def owner : Owner := ⟨.program ⟨257⟩, ⟨23486⟩⟩
def mergeEvent : Nat := 171123
def frameStart : Nat := 171030
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171118RawTerms
def rightRaw : List Term := Proof.Events668.exact171075RawTerms
def group : MergeGroup := .operator 171118 171075
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171118) (leftOrdinal := 1)
    (rightResult := 171075) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23483⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171123

namespace LeftMerge171125
def owner : Owner := ⟨.program ⟨257⟩, ⟨23486⟩⟩
def mergeEvent : Nat := 171125
def frameStart : Nat := 171030
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22953⟩⟩] } }
def rhsRaw : List Term := Proof.Events668.exact171072RawTerms
def group : MergeGroup := .relation 171124
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171124) (rhsResult := 171072)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23483⟩⟩) ⟨22953⟩ 171072) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22953⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], [⟨.program ⟨257⟩, ⟨22953⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171125

namespace LeftMerge171133
def owner : Owner := ⟨.program ⟨257⟩, ⟨21842⟩⟩
def mergeEvent : Nat := 171133
def frameStart : Nat := 171030
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171086RawTerms
def rightRaw : List Term := Proof.Events668.exact171129RawTerms
def group : MergeGroup := .operator 171086 171129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171086) (leftOrdinal := 0)
    (rightResult := 171129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171133

namespace LeftMerge171150
def owner : Owner := ⟨.program ⟨257⟩, ⟨22412⟩⟩
def mergeEvent : Nat := 171150
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }
def rhsRaw : List Term := Proof.Events668.exact171147RawTerms
def group : MergeGroup := .relation 171149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171149) (rhsResult := 171147)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171148 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩]⟩) (none) 171147) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171150

namespace LeftMerge171151
def owner : Owner := ⟨.program ⟨257⟩, ⟨22412⟩⟩
def mergeEvent : Nat := 171151
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩] } }
def rhsRaw : List Term := Proof.Events668.exact171147RawTerms
def group : MergeGroup := .relation 171149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171149) (rhsResult := 171147)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171148 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩]⟩) (none) 171147) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171151

namespace LeftMerge171152
def owner : Owner := ⟨.program ⟨257⟩, ⟨22412⟩⟩
def mergeEvent : Nat := 171152
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22953⟩⟩] } }
def rhsRaw : List Term := Proof.Events668.exact171147RawTerms
def group : MergeGroup := .relation 171149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171149) (rhsResult := 171147)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171148 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩]⟩) (none) 171147) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22953⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], [⟨.program ⟨257⟩, ⟨22953⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171152

namespace LeftMerge171153
def owner : Owner := ⟨.program ⟨257⟩, ⟨22412⟩⟩
def mergeEvent : Nat := 171153
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events668.exact171147RawTerms
def group : MergeGroup := .relation 171149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 171149) (rhsResult := 171147)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 171148 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22409⟩⟩]⟩) (none) 171147) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21840⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21840⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171153

namespace LeftMerge171158
def owner : Owner := ⟨.program ⟨257⟩, ⟨23485⟩⟩
def mergeEvent : Nat := 171158
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22953⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171154RawTerms
def rightRaw : List Term := Proof.Events667.exact170968RawTerms
def group : MergeGroup := .operator 171154 170968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171154) (leftOrdinal := 2)
    (rightResult := 170968) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22953⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22953⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨21161⟩⟩, ⟨.program ⟨257⟩, ⟨21590⟩⟩], [⟨.program ⟨257⟩, ⟨22953⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge171158

namespace LeftMerge171159
def owner : Owner := ⟨.program ⟨257⟩, ⟨23485⟩⟩
def mergeEvent : Nat := 171159
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩] } }
def leftRaw : List Term := Proof.Events668.exact171154RawTerms
def rightRaw : List Term := Proof.Events667.exact170968RawTerms
def group : MergeGroup := .operator 171154 170968
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 171154) (leftOrdinal := 1)
    (rightResult := 170968) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23483⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge171159

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
