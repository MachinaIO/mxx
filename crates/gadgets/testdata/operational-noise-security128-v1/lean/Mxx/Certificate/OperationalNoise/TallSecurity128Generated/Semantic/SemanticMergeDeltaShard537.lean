import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge90257
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16137RawTerms
def group : MergeGroup := .relation 90256
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90256) (rhsResult := 16137)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9499⟩⟩) ⟨7243⟩ 16137) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨51279⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90257

namespace LeftMerge90258
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90258
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90207RawTerms
def rightRaw : List Term := Proof.Events063.exact16144RawTerms
def group : MergeGroup := .operator 90207 16144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90207) (leftOrdinal := 7)
    (rightResult := 16144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90258

namespace LeftMerge90260
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90260
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16137RawTerms
def group : MergeGroup := .relation 90259
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90259) (rhsResult := 16137)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9499⟩⟩) ⟨7243⟩ 16137) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨32215⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90260

namespace LeftMerge90261
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90261
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90207RawTerms
def rightRaw : List Term := Proof.Events063.exact16144RawTerms
def group : MergeGroup := .operator 90207 16144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90207) (leftOrdinal := 11)
    (rightResult := 16144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90261

namespace LeftMerge90263
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90263
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16137RawTerms
def group : MergeGroup := .relation 90262
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90262) (rhsResult := 16137)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9499⟩⟩) ⟨7243⟩ 16137) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6822⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22195⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90263

namespace LeftMerge90264
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90264
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90207RawTerms
def rightRaw : List Term := Proof.Events063.exact16144RawTerms
def group : MergeGroup := .operator 90207 16144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90207) (leftOrdinal := 15)
    (rightResult := 16144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90264

namespace LeftMerge90266
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90266
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16137RawTerms
def group : MergeGroup := .relation 90265
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90265) (rhsResult := 16137)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9499⟩⟩) ⟨7243⟩ 16137) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6846⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨18975⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90266

namespace LeftMerge90267
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90267
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90207RawTerms
def rightRaw : List Term := Proof.Events063.exact16144RawTerms
def group : MergeGroup := .operator 90207 16144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90207) (leftOrdinal := 18)
    (rightResult := 16144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90267

namespace LeftMerge90269
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90269
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def rhsRaw : List Term := Proof.Events063.exact16137RawTerms
def group : MergeGroup := .relation 90268
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 90268) (rhsResult := 16137)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9499⟩⟩) ⟨7243⟩ 16137) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6863⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨16126⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90269

namespace LeftMerge90270
def owner : Owner := ⟨.program ⟨257⟩, ⟨71445⟩⟩
def mergeEvent : Nat := 90270
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90207RawTerms
def rightRaw : List Term := Proof.Events063.exact16144RawTerms
def group : MergeGroup := .operator 90207 16144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90207) (leftOrdinal := 0)
    (rightResult := 16144) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9499⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7242⟩⟩, ⟨.program ⟨257⟩, ⟨9583⟩⟩, ⟨.program ⟨257⟩, ⟨9499⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90270

namespace LeftMerge90275
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90275
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 6)
    (rightResult := 75868) (rightOrdinal := 24) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6774⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨67586⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge90275

namespace LeftMerge90276
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 8)
    (rightResult := 75868) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨48437⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90276

namespace LeftMerge90277
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90277
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 9)
    (rightResult := 75868) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6807⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨45757⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90277

namespace LeftMerge90278
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90278
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 10)
    (rightResult := 75868) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6817⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨43080⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90278

namespace LeftMerge90279
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90279
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 12)
    (rightResult := 75868) (rightOrdinal := 30) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6828⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨40400⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90279

namespace LeftMerge90280
def owner : Owner := ⟨.program ⟨257⟩, ⟨71446⟩⟩
def mergeEvent : Nat := 90280
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }
def leftRaw : List Term := Proof.Events352.exact90271RawTerms
def rightRaw : List Term := Proof.Events296.exact75868RawTerms
def group : MergeGroup := .operator 90271 75868
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90271) (leftOrdinal := 13)
    (rightResult := 75868) (rightOrdinal := 31) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7243⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨7243⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge90280

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
