import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge47839
def owner : Owner := ⟨.program ⟨214⟩, ⟨28538⟩⟩
def mergeEvent : Nat := 47839
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39697RawTerms
def rightRaw : List Term := Proof.Events186.exact47833RawTerms
def group : MergeGroup := .operator 39697 47833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39697) (leftOrdinal := 0)
    (rightResult := 47833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28536⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge47839

namespace LeftMerge47840
def owner : Owner := ⟨.program ⟨214⟩, ⟨28538⟩⟩
def mergeEvent : Nat := 47840
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩] } }
def leftRaw : List Term := Proof.Events155.exact39697RawTerms
def rightRaw : List Term := Proof.Events186.exact47833RawTerms
def group : MergeGroup := .operator 39697 47833
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 39697) (leftOrdinal := 1)
    (rightResult := 47833) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28536⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge47840

namespace LeftMerge47842
def owner : Owner := ⟨.program ⟨214⟩, ⟨28538⟩⟩
def mergeEvent : Nat := 47842
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24356⟩⟩] } }
def rhsRaw : List Term := Proof.Events186.exact47830RawTerms
def group : MergeGroup := .relation 47841
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 47841) (rhsResult := 47830)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28536⟩⟩) ⟨24356⟩ 47830) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24356⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge47842

namespace LeftMerge47856
def owner : Owner := ⟨.program ⟨214⟩, ⟨21771⟩⟩
def mergeEvent : Nat := 47856
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events186.exact47850RawTerms
def group : MergeGroup := .operator 36137 47850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 47850) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21768⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge47856

namespace LeftMerge47977
def owner : Owner := ⟨.program ⟨214⟩, ⟨16347⟩⟩
def mergeEvent : Nat := 47977
def frameStart : Nat := 47911
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact47973RawTerms
def rightRaw : List Term := Proof.Events187.exact47971RawTerms
def group : MergeGroup := .operator 47973 47971
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 47973) (leftOrdinal := 0)
    (rightResult := 47971) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge47977

namespace LeftMerge47989
def owner : Owner := ⟨.program ⟨214⟩, ⟨28537⟩⟩
def mergeEvent : Nat := 47989
def frameStart : Nat := 47911
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact47985RawTerms
def rightRaw : List Term := Proof.Events187.exact47962RawTerms
def group : MergeGroup := .operator 47985 47962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 47985) (leftOrdinal := 0)
    (rightResult := 47962) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28536⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge47989

namespace LeftMerge47990
def owner : Owner := ⟨.program ⟨214⟩, ⟨28537⟩⟩
def mergeEvent : Nat := 47990
def frameStart : Nat := 47911
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact47985RawTerms
def rightRaw : List Term := Proof.Events187.exact47962RawTerms
def group : MergeGroup := .operator 47985 47962
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 47985) (leftOrdinal := 1)
    (rightResult := 47962) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28536⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge47990

namespace LeftMerge47992
def owner : Owner := ⟨.program ⟨214⟩, ⟨28537⟩⟩
def mergeEvent : Nat := 47992
def frameStart : Nat := 47911
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24356⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact47959RawTerms
def group : MergeGroup := .relation 47991
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 47991) (rhsResult := 47959)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28536⟩⟩) ⟨24356⟩ 47959) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24356⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge47992

namespace LeftMerge48000
def owner : Owner := ⟨.program ⟨214⟩, ⟨17616⟩⟩
def mergeEvent : Nat := 48000
def frameStart : Nat := 47911
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact47973RawTerms
def rightRaw : List Term := Proof.Events187.exact47996RawTerms
def group : MergeGroup := .operator 47973 47996
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 47973) (leftOrdinal := 0)
    (rightResult := 47996) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48000

namespace LeftMerge48017
def owner : Owner := ⟨.program ⟨214⟩, ⟨21771⟩⟩
def mergeEvent : Nat := 48017
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact48014RawTerms
def group : MergeGroup := .relation 48016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48016) (rhsResult := 48014)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48015 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩) (none) 48014) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48017

namespace LeftMerge48018
def owner : Owner := ⟨.program ⟨214⟩, ⟨21771⟩⟩
def mergeEvent : Nat := 48018
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact48014RawTerms
def group : MergeGroup := .relation 48016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48016) (rhsResult := 48014)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48015 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩) (none) 48014) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48018

namespace LeftMerge48019
def owner : Owner := ⟨.program ⟨214⟩, ⟨21771⟩⟩
def mergeEvent : Nat := 48019
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24356⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact48014RawTerms
def group : MergeGroup := .relation 48016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48016) (rhsResult := 48014)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48015 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩) (none) 48014) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24356⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48019

namespace LeftMerge48020
def owner : Owner := ⟨.program ⟨214⟩, ⟨21771⟩⟩
def mergeEvent : Nat := 48020
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events187.exact48014RawTerms
def group : MergeGroup := .relation 48016
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 48016) (rhsResult := 48014)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 48015 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21768⟩⟩]⟩) (none) 48014) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17614⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17614⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48020

namespace LeftMerge48025
def owner : Owner := ⟨.program ⟨214⟩, ⟨28539⟩⟩
def mergeEvent : Nat := 48025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact48021RawTerms
def rightRaw : List Term := Proof.Events186.exact47843RawTerms
def group : MergeGroup := .operator 48021 47843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48021) (leftOrdinal := 0)
    (rightResult := 47843) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6700⟩⟩, ⟨.program ⟨214⟩, ⟨28536⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48025

namespace LeftMerge48026
def owner : Owner := ⟨.program ⟨214⟩, ⟨28539⟩⟩
def mergeEvent : Nat := 48026
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24356⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact48021RawTerms
def rightRaw : List Term := Proof.Events186.exact47843RawTerms
def group : MergeGroup := .operator 48021 47843
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48021) (leftOrdinal := 2)
    (rightResult := 47843) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24356⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24356⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16270⟩⟩], [⟨.program ⟨214⟩, ⟨24356⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge48026

namespace LeftMerge48034
def owner : Owner := ⟨.program ⟨214⟩, ⟨28540⟩⟩
def mergeEvent : Nat := 48034
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩] } }
def leftRaw : List Term := Proof.Events187.exact48028RawTerms
def rightRaw : List Term := Proof.Events022.exact5659RawTerms
def group : MergeGroup := .operator 48028 5659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 48028) (leftOrdinal := 0)
    (rightResult := 5659) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6728⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6677⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6728⟩⟩, ⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge48034

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
