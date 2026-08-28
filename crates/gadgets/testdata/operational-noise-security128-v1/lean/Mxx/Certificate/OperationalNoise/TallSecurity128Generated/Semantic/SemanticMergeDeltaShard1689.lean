import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge274008
def owner : Owner := ⟨.program ⟨257⟩, ⟨19069⟩⟩
def mergeEvent : Nat := 274008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20128⟩⟩] } }
def rhsRaw : List Term := Proof.Events1070.exact274004RawTerms
def group : MergeGroup := .relation 274006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274006) (rhsResult := 274004)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19066⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 274005 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19066⟩⟩]⟩) (none) 274004) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20128⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20128⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274008

namespace LeftMerge274009
def owner : Owner := ⟨.program ⟨257⟩, ⟨19069⟩⟩
def mergeEvent : Nat := 274009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12556⟩⟩, ⟨.program ⟨257⟩, ⟨18074⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19659⟩⟩] } }
def rhsRaw : List Term := Proof.Events1070.exact274004RawTerms
def group : MergeGroup := .relation 274006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274006) (rhsResult := 274004)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19066⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 274005 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19066⟩⟩]⟩) (none) 274004) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12556⟩⟩, ⟨.program ⟨257⟩, ⟨18074⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19659⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12556⟩⟩, ⟨.program ⟨257⟩, ⟨18074⟩⟩], [⟨.program ⟨257⟩, ⟨19659⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274009

namespace LeftMerge274010
def owner : Owner := ⟨.program ⟨257⟩, ⟨19069⟩⟩
def mergeEvent : Nat := 274010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1070.exact274004RawTerms
def group : MergeGroup := .relation 274006
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274006) (rhsResult := 274004)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19066⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 274005 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19066⟩⟩]⟩) (none) 274004) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274010

namespace LeftMerge274015
def owner : Owner := ⟨.program ⟨257⟩, ⟨20130⟩⟩
def mergeEvent : Nat := 274015
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12556⟩⟩, ⟨.program ⟨257⟩, ⟨18074⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19659⟩⟩] } }
def leftRaw : List Term := Proof.Events1070.exact274011RawTerms
def rightRaw : List Term := Proof.Events1069.exact273825RawTerms
def group : MergeGroup := .operator 274011 273825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274011) (leftOrdinal := 2)
    (rightResult := 273825) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12556⟩⟩, ⟨.program ⟨257⟩, ⟨18074⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19659⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12556⟩⟩, ⟨.program ⟨257⟩, ⟨18074⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19659⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12556⟩⟩, ⟨.program ⟨257⟩, ⟨18074⟩⟩], [⟨.program ⟨257⟩, ⟨19659⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274015

namespace LeftMerge274016
def owner : Owner := ⟨.program ⟨257⟩, ⟨20130⟩⟩
def mergeEvent : Nat := 274016
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20128⟩⟩] } }
def leftRaw : List Term := Proof.Events1070.exact274011RawTerms
def rightRaw : List Term := Proof.Events1069.exact273825RawTerms
def group : MergeGroup := .operator 274011 273825
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274011) (leftOrdinal := 1)
    (rightResult := 273825) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20128⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20128⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20128⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274016

namespace LeftMerge274024
def owner : Owner := ⟨.program ⟨257⟩, ⟨20397⟩⟩
def mergeEvent : Nat := 274024
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩] } }
def leftRaw : List Term := Proof.Events1070.exact274018RawTerms
def rightRaw : List Term := Proof.Events1069.exact273741RawTerms
def group : MergeGroup := .operator 274018 273741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274018) (leftOrdinal := 0)
    (rightResult := 273741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274024

namespace LeftMerge274025
def owner : Owner := ⟨.program ⟨257⟩, ⟨20397⟩⟩
def mergeEvent : Nat := 274025
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩] } }
def leftRaw : List Term := Proof.Events1070.exact274018RawTerms
def rightRaw : List Term := Proof.Events1069.exact273741RawTerms
def group : MergeGroup := .operator 274018 273741
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274018) (leftOrdinal := 1)
    (rightResult := 273741) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274025

namespace LeftMerge274027
def owner : Owner := ⟨.program ⟨257⟩, ⟨20397⟩⟩
def mergeEvent : Nat := 274027
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19786⟩⟩] } }
def rhsRaw : List Term := Proof.Events1069.exact273738RawTerms
def group : MergeGroup := .relation 274026
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274026) (rhsResult := 273738)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20395⟩⟩) ⟨19786⟩ 273738) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19786⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨18522⟩⟩], [⟨.program ⟨257⟩, ⟨19786⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274027

namespace LeftMerge274041
def owner : Owner := ⟨.program ⟨257⟩, ⟨19293⟩⟩
def mergeEvent : Nat := 274041
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19290⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1070.exact274035RawTerms
def group : MergeGroup := .operator 266120 274035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 274035) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19290⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19290⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274041

namespace LeftMerge274162
def owner : Owner := ⟨.program ⟨257⟩, ⟨20036⟩⟩
def mergeEvent : Nat := 274162
def frameStart : Nat := 274096
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1070.exact274158RawTerms
def rightRaw : List Term := Proof.Events1070.exact274156RawTerms
def group : MergeGroup := .operator 274158 274156
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274158) (leftOrdinal := 0)
    (rightResult := 274156) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274162

namespace LeftMerge274174
def owner : Owner := ⟨.program ⟨257⟩, ⟨20396⟩⟩
def mergeEvent : Nat := 274174
def frameStart : Nat := 274096
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩] } }
def leftRaw : List Term := Proof.Events1070.exact274170RawTerms
def rightRaw : List Term := Proof.Events1070.exact274147RawTerms
def group : MergeGroup := .operator 274170 274147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274170) (leftOrdinal := 0)
    (rightResult := 274147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20395⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274174

namespace LeftMerge274175
def owner : Owner := ⟨.program ⟨257⟩, ⟨20396⟩⟩
def mergeEvent : Nat := 274175
def frameStart : Nat := 274096
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩] } }
def leftRaw : List Term := Proof.Events1070.exact274170RawTerms
def rightRaw : List Term := Proof.Events1070.exact274147RawTerms
def group : MergeGroup := .operator 274170 274147
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274170) (leftOrdinal := 1)
    (rightResult := 274147) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20395⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274175

namespace LeftMerge274177
def owner : Owner := ⟨.program ⟨257⟩, ⟨20396⟩⟩
def mergeEvent : Nat := 274177
def frameStart : Nat := 274096
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18522⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19786⟩⟩] } }
def rhsRaw : List Term := Proof.Events1070.exact274144RawTerms
def group : MergeGroup := .relation 274176
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274176) (rhsResult := 274144)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18522⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20395⟩⟩) ⟨19786⟩ 274144) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19786⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18522⟩⟩], [⟨.program ⟨257⟩, ⟨19786⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274177

namespace LeftMerge274185
def owner : Owner := ⟨.program ⟨257⟩, ⟨18711⟩⟩
def mergeEvent : Nat := 274185
def frameStart : Nat := 274096
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18709⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1070.exact274158RawTerms
def rightRaw : List Term := Proof.Events1071.exact274181RawTerms
def group : MergeGroup := .operator 274158 274181
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 274158) (leftOrdinal := 0)
    (rightResult := 274181) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18709⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18709⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274185

namespace LeftMerge274202
def owner : Owner := ⟨.program ⟨257⟩, ⟨19293⟩⟩
def mergeEvent : Nat := 274202
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }
def rhsRaw : List Term := Proof.Events1071.exact274199RawTerms
def group : MergeGroup := .relation 274201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274201) (rhsResult := 274199)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19290⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 274200 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19290⟩⟩]⟩) (none) 274199) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge274202

namespace LeftMerge274203
def owner : Owner := ⟨.program ⟨257⟩, ⟨19293⟩⟩
def mergeEvent : Nat := 274203
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩] } }
def rhsRaw : List Term := Proof.Events1071.exact274199RawTerms
def group : MergeGroup := .relation 274201
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 274201) (rhsResult := 274199)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19290⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 274200 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19290⟩⟩]⟩) (none) 274199) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20395⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge274203

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
