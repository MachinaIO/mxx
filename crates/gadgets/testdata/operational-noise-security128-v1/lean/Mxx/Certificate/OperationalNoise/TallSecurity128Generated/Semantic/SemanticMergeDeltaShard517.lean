import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge87212
def owner : Owner := ⟨.program ⟨257⟩, ⟨39454⟩⟩
def mergeEvent : Nat := 87212
def frameStart : Nat := 87133
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩] } }
def leftRaw : List Term := Proof.Events340.exact87207RawTerms
def rightRaw : List Term := Proof.Events340.exact87184RawTerms
def group : MergeGroup := .operator 87207 87184
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87207) (leftOrdinal := 1)
    (rightResult := 87184) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨39453⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87212

namespace LeftMerge87214
def owner : Owner := ⟨.program ⟨257⟩, ⟨39454⟩⟩
def mergeEvent : Nat := 87214
def frameStart : Nat := 87133
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38634⟩⟩] } }
def rhsRaw : List Term := Proof.Events340.exact87181RawTerms
def group : MergeGroup := .relation 87213
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87213) (rhsResult := 87181)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨39453⟩⟩) ⟨38634⟩ 87181) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨38634⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38634⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87214

namespace LeftMerge87222
def owner : Owner := ⟨.program ⟨257⟩, ⟨37719⟩⟩
def mergeEvent : Nat := 87222
def frameStart : Nat := 87133
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events340.exact87195RawTerms
def rightRaw : List Term := Proof.Events340.exact87218RawTerms
def group : MergeGroup := .operator 87195 87218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87195) (leftOrdinal := 0)
    (rightResult := 87218) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87222

namespace LeftMerge87239
def owner : Owner := ⟨.program ⟨257⟩, ⟨38295⟩⟩
def mergeEvent : Nat := 87239
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩] } }
def rhsRaw : List Term := Proof.Events340.exact87236RawTerms
def group : MergeGroup := .relation 87238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87238) (rhsResult := 87236)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38292⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87237 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38292⟩⟩]⟩) (none) 87236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87239

namespace LeftMerge87240
def owner : Owner := ⟨.program ⟨257⟩, ⟨38295⟩⟩
def mergeEvent : Nat := 87240
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩] } }
def rhsRaw : List Term := Proof.Events340.exact87236RawTerms
def group : MergeGroup := .relation 87238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87238) (rhsResult := 87236)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38292⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87237 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38292⟩⟩]⟩) (none) 87236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87240

namespace LeftMerge87241
def owner : Owner := ⟨.program ⟨257⟩, ⟨38295⟩⟩
def mergeEvent : Nat := 87241
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38634⟩⟩] } }
def rhsRaw : List Term := Proof.Events340.exact87236RawTerms
def group : MergeGroup := .relation 87238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87238) (rhsResult := 87236)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38292⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87237 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38292⟩⟩]⟩) (none) 87236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38634⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38634⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87241

namespace LeftMerge87242
def owner : Owner := ⟨.program ⟨257⟩, ⟨38295⟩⟩
def mergeEvent : Nat := 87242
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events340.exact87236RawTerms
def group : MergeGroup := .relation 87238
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87238) (rhsResult := 87236)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38292⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87237 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨38292⟩⟩]⟩) (none) 87236) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87242

namespace LeftMerge87247
def owner : Owner := ⟨.program ⟨257⟩, ⟨39456⟩⟩
def mergeEvent : Nat := 87247
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩] } }
def leftRaw : List Term := Proof.Events340.exact87243RawTerms
def rightRaw : List Term := Proof.Events340.exact87065RawTerms
def group : MergeGroup := .operator 87243 87065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87243) (leftOrdinal := 0)
    (rightResult := 87065) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7192⟩⟩, ⟨.program ⟨257⟩, ⟨39453⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87247

namespace LeftMerge87248
def owner : Owner := ⟨.program ⟨257⟩, ⟨39456⟩⟩
def mergeEvent : Nat := 87248
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38634⟩⟩] } }
def leftRaw : List Term := Proof.Events340.exact87243RawTerms
def rightRaw : List Term := Proof.Events340.exact87065RawTerms
def group : MergeGroup := .operator 87243 87065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87243) (leftOrdinal := 2)
    (rightResult := 87065) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38634⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38634⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37476⟩⟩], [⟨.program ⟨257⟩, ⟨38634⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87248

namespace LeftMerge87256
def owner : Owner := ⟨.program ⟨257⟩, ⟨39457⟩⟩
def mergeEvent : Nat := 87256
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩] } }
def leftRaw : List Term := Proof.Events340.exact87250RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 87250 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87250) (leftOrdinal := 0)
    (rightResult := 15622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7161⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87256

namespace LeftMerge87257
def owner : Owner := ⟨.program ⟨257⟩, ⟨39457⟩⟩
def mergeEvent : Nat := 87257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩] } }
def leftRaw : List Term := Proof.Events340.exact87250RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 87250 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87250) (leftOrdinal := 1)
    (rightResult := 15622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7161⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87257

namespace LeftMerge87259
def owner : Owner := ⟨.program ⟨257⟩, ⟨39457⟩⟩
def mergeEvent : Nat := 87259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15615RawTerms
def group : MergeGroup := .relation 87258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87258) (rhsResult := 15615)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7161⟩⟩) ⟨7046⟩ 15615) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨37717⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87259

namespace LeftMerge87273
def owner : Owner := ⟨.program ⟨257⟩, ⟨36775⟩⟩
def mergeEvent : Nat := 87273
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩] } }
def leftRaw : List Term := Proof.Events306.exact78591RawTerms
def rightRaw : List Term := Proof.Events340.exact87267RawTerms
def group : MergeGroup := .operator 78591 87267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78591) (leftOrdinal := 0)
    (rightResult := 87267) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36773⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87273

namespace LeftMerge87274
def owner : Owner := ⟨.program ⟨257⟩, ⟨36775⟩⟩
def mergeEvent : Nat := 87274
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩] } }
def leftRaw : List Term := Proof.Events306.exact78591RawTerms
def rightRaw : List Term := Proof.Events340.exact87267RawTerms
def group : MergeGroup := .operator 78591 87267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 78591) (leftOrdinal := 1)
    (rightResult := 87267) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36773⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87274

namespace LeftMerge87276
def owner : Owner := ⟨.program ⟨257⟩, ⟨36775⟩⟩
def mergeEvent : Nat := 87276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35954⟩⟩] } }
def rhsRaw : List Term := Proof.Events340.exact87264RawTerms
def group : MergeGroup := .relation 87275
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87275) (rhsResult := 87264)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36773⟩⟩) ⟨35954⟩ 87264) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35954⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87276

namespace LeftMerge87290
def owner : Owner := ⟨.program ⟨257⟩, ⟨35615⟩⟩
def mergeEvent : Nat := 87290
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events340.exact87284RawTerms
def group : MergeGroup := .operator 75995 87284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 87284) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35612⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87290

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
