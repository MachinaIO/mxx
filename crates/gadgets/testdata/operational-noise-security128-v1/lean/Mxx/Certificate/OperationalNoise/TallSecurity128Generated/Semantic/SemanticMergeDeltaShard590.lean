import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge98992
def owner : Owner := ⟨.program ⟨257⟩, ⟨16342⟩⟩
def mergeEvent : Nat := 98992
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events386.exact98986RawTerms
def group : MergeGroup := .relation 98988
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 98988) (rhsResult := 98986)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 98987 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16339⟩⟩]⟩) (none) 98986) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98992

namespace LeftMerge98997
def owner : Owner := ⟨.program ⟨257⟩, ⟨17416⟩⟩
def mergeEvent : Nat := 98997
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16879⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98993RawTerms
def rightRaw : List Term := Proof.Events385.exact98807RawTerms
def group : MergeGroup := .operator 98993 98807
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98993) (leftOrdinal := 2)
    (rightResult := 98807) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16879⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16879⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨12456⟩⟩, ⟨.program ⟨257⟩, ⟨15594⟩⟩], [⟨.program ⟨257⟩, ⟨16879⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge98997

namespace LeftMerge98998
def owner : Owner := ⟨.program ⟨257⟩, ⟨17416⟩⟩
def mergeEvent : Nat := 98998
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact98993RawTerms
def rightRaw : List Term := Proof.Events385.exact98807RawTerms
def group : MergeGroup := .operator 98993 98807
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 98993) (leftOrdinal := 1)
    (rightResult := 98807) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17414⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge98998

namespace LeftMerge99006
def owner : Owner := ⟨.program ⟨257⟩, ⟨17903⟩⟩
def mergeEvent : Nat := 99006
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact99000RawTerms
def rightRaw : List Term := Proof.Events385.exact98723RawTerms
def group : MergeGroup := .operator 99000 98723
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99000) (leftOrdinal := 0)
    (rightResult := 98723) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17901⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99006

namespace LeftMerge99007
def owner : Owner := ⟨.program ⟨257⟩, ⟨17903⟩⟩
def mergeEvent : Nat := 99007
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩] } }
def leftRaw : List Term := Proof.Events386.exact99000RawTerms
def rightRaw : List Term := Proof.Events385.exact98723RawTerms
def group : MergeGroup := .operator 99000 98723
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99000) (leftOrdinal := 1)
    (rightResult := 98723) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17901⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99007

namespace LeftMerge99009
def owner : Owner := ⟨.program ⟨257⟩, ⟨17903⟩⟩
def mergeEvent : Nat := 99009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }
def rhsRaw : List Term := Proof.Events385.exact98720RawTerms
def group : MergeGroup := .relation 99008
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99008) (rhsResult := 98720)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17901⟩⟩) ⟨17046⟩ 98720) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17046⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99009

namespace LeftMerge99023
def owner : Owner := ⟨.program ⟨257⟩, ⟨16699⟩⟩
def mergeEvent : Nat := 99023
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events386.exact99017RawTerms
def group : MergeGroup := .operator 90620 99017
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 99017) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16696⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99023

namespace LeftMerge99144
def owner : Owner := ⟨.program ⟨257⟩, ⟨17228⟩⟩
def mergeEvent : Nat := 99144
def frameStart : Nat := 99078
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99140RawTerms
def rightRaw : List Term := Proof.Events387.exact99138RawTerms
def group : MergeGroup := .operator 99140 99138
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99140) (leftOrdinal := 0)
    (rightResult := 99138) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99144

namespace LeftMerge99156
def owner : Owner := ⟨.program ⟨257⟩, ⟨17902⟩⟩
def mergeEvent : Nat := 99156
def frameStart : Nat := 99078
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99152RawTerms
def rightRaw : List Term := Proof.Events387.exact99129RawTerms
def group : MergeGroup := .operator 99152 99129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99152) (leftOrdinal := 0)
    (rightResult := 99129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17901⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99156

namespace LeftMerge99157
def owner : Owner := ⟨.program ⟨257⟩, ⟨17902⟩⟩
def mergeEvent : Nat := 99157
def frameStart : Nat := 99078
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99152RawTerms
def rightRaw : List Term := Proof.Events387.exact99129RawTerms
def group : MergeGroup := .operator 99152 99129
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99152) (leftOrdinal := 1)
    (rightResult := 99129) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17901⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99157

namespace LeftMerge99159
def owner : Owner := ⟨.program ⟨257⟩, ⟨17902⟩⟩
def mergeEvent : Nat := 99159
def frameStart : Nat := 99078
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }
def rhsRaw : List Term := Proof.Events387.exact99126RawTerms
def group : MergeGroup := .relation 99158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99158) (rhsResult := 99126)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17901⟩⟩) ⟨17046⟩ 99126) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17046⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99159

namespace LeftMerge99167
def owner : Owner := ⟨.program ⟨257⟩, ⟨16116⟩⟩
def mergeEvent : Nat := 99167
def frameStart : Nat := 99078
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16115⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events387.exact99140RawTerms
def rightRaw : List Term := Proof.Events387.exact99163RawTerms
def group : MergeGroup := .operator 99140 99163
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 99140) (leftOrdinal := 0)
    (rightResult := 99163) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16115⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16115⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99167

namespace LeftMerge99184
def owner : Owner := ⟨.program ⟨257⟩, ⟨16699⟩⟩
def mergeEvent : Nat := 99184
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }
def rhsRaw : List Term := Proof.Events387.exact99181RawTerms
def group : MergeGroup := .relation 99183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99183) (rhsResult := 99181)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99182 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩) (none) 99181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99184

namespace LeftMerge99185
def owner : Owner := ⟨.program ⟨257⟩, ⟨16699⟩⟩
def mergeEvent : Nat := 99185
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩] } }
def rhsRaw : List Term := Proof.Events387.exact99181RawTerms
def group : MergeGroup := .relation 99183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99183) (rhsResult := 99181)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99182 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩) (none) 99181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17901⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99185

namespace LeftMerge99186
def owner : Owner := ⟨.program ⟨257⟩, ⟨16699⟩⟩
def mergeEvent : Nat := 99186
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }
def rhsRaw : List Term := Proof.Events387.exact99181RawTerms
def group : MergeGroup := .relation 99183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99183) (rhsResult := 99181)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99182 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩) (none) 99181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨17046⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨15828⟩⟩], [⟨.program ⟨257⟩, ⟨17046⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge99186

namespace LeftMerge99187
def owner : Owner := ⟨.program ⟨257⟩, ⟨16699⟩⟩
def mergeEvent : Nat := 99187
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16115⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events387.exact99181RawTerms
def group : MergeGroup := .relation 99183
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 99183) (rhsResult := 99181)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 99182 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16696⟩⟩]⟩) (none) 99181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16115⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨16115⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge99187

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
