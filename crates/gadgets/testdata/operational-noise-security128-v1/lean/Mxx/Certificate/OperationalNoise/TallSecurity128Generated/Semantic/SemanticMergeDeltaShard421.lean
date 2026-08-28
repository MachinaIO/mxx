import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge71768
def owner : Owner := ⟨.program ⟨257⟩, ⟨49035⟩⟩
def mergeEvent : Nat := 71768
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49363⟩⟩] } }
def rhsRaw : List Term := Proof.Events280.exact71763RawTerms
def group : MergeGroup := .relation 71765
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71765) (rhsResult := 71763)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 71764 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩) (none) 71763) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49363⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71768

namespace LeftMerge71769
def owner : Owner := ⟨.program ⟨257⟩, ⟨49035⟩⟩
def mergeEvent : Nat := 71769
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events280.exact71763RawTerms
def group : MergeGroup := .relation 71765
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71765) (rhsResult := 71763)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 71764 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨49032⟩⟩]⟩) (none) 71763) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71769

namespace LeftMerge71774
def owner : Owner := ⟨.program ⟨257⟩, ⟨50201⟩⟩
def mergeEvent : Nat := 71774
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71770RawTerms
def rightRaw : List Term := Proof.Events279.exact71592RawTerms
def group : MergeGroup := .operator 71770 71592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71770) (leftOrdinal := 0)
    (rightResult := 71592) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71774

namespace LeftMerge71775
def owner : Owner := ⟨.program ⟨257⟩, ⟨50201⟩⟩
def mergeEvent : Nat := 71775
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49363⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71770RawTerms
def rightRaw : List Term := Proof.Events279.exact71592RawTerms
def group : MergeGroup := .operator 71770 71592
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71770) (leftOrdinal := 2)
    (rightResult := 71592) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49363⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49363⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨49363⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71775

namespace LeftMerge71783
def owner : Owner := ⟨.program ⟨257⟩, ⟨50202⟩⟩
def mergeEvent : Nat := 71783
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71777RawTerms
def rightRaw : List Term := Proof.Events060.exact15542RawTerms
def group : MergeGroup := .operator 71777 15542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71777) (leftOrdinal := 0)
    (rightResult := 15542) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7231⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7147⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7231⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71783

namespace LeftMerge71784
def owner : Owner := ⟨.program ⟨257⟩, ⟨50202⟩⟩
def mergeEvent : Nat := 71784
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71777RawTerms
def rightRaw : List Term := Proof.Events060.exact15542RawTerms
def group : MergeGroup := .operator 71777 15542
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71777) (leftOrdinal := 1)
    (rightResult := 15542) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7147⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71784

namespace LeftMerge71786
def owner : Owner := ⟨.program ⟨257⟩, ⟨50202⟩⟩
def mergeEvent : Nat := 71786
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15535RawTerms
def group : MergeGroup := .relation 71785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71785) (rhsResult := 15535)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7147⟩⟩) ⟨7039⟩ 15535) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6800⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6800⟩⟩, ⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48450⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71786

namespace LeftMerge71800
def owner : Owner := ⟨.program ⟨257⟩, ⟨47520⟩⟩
def mergeEvent : Nat := 71800
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62038RawTerms
def rightRaw : List Term := Proof.Events280.exact71794RawTerms
def group : MergeGroup := .operator 62038 71794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62038) (leftOrdinal := 0)
    (rightResult := 71794) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47518⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71800

namespace LeftMerge71801
def owner : Owner := ⟨.program ⟨257⟩, ⟨47520⟩⟩
def mergeEvent : Nat := 71801
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩] } }
def leftRaw : List Term := Proof.Events242.exact62038RawTerms
def rightRaw : List Term := Proof.Events280.exact71794RawTerms
def group : MergeGroup := .operator 62038 71794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 62038) (leftOrdinal := 1)
    (rightResult := 71794) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47518⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71801

namespace LeftMerge71803
def owner : Owner := ⟨.program ⟨257⟩, ⟨47520⟩⟩
def mergeEvent : Nat := 71803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46683⟩⟩] } }
def rhsRaw : List Term := Proof.Events280.exact71791RawTerms
def group : MergeGroup := .relation 71802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71802) (rhsResult := 71791)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47518⟩⟩) ⟨46683⟩ 71791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46683⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨46683⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71803

namespace LeftMerge71817
def owner : Owner := ⟨.program ⟨257⟩, ⟨46355⟩⟩
def mergeEvent : Nat := 71817
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46352⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events280.exact71811RawTerms
def group : MergeGroup := .operator 61370 71811
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 71811) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46352⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46352⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71817

namespace LeftMerge71938
def owner : Owner := ⟨.program ⟨257⟩, ⟨46856⟩⟩
def mergeEvent : Nat := 71938
def frameStart : Nat := 71872
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71934RawTerms
def rightRaw : List Term := Proof.Events280.exact71932RawTerms
def group : MergeGroup := .operator 71934 71932
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71934) (leftOrdinal := 0)
    (rightResult := 71932) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71938

namespace LeftMerge71950
def owner : Owner := ⟨.program ⟨257⟩, ⟨47519⟩⟩
def mergeEvent : Nat := 71950
def frameStart : Nat := 71872
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩] } }
def leftRaw : List Term := Proof.Events281.exact71946RawTerms
def rightRaw : List Term := Proof.Events280.exact71923RawTerms
def group : MergeGroup := .operator 71946 71923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71946) (leftOrdinal := 0)
    (rightResult := 71923) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47518⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71950

namespace LeftMerge71951
def owner : Owner := ⟨.program ⟨257⟩, ⟨47519⟩⟩
def mergeEvent : Nat := 71951
def frameStart : Nat := 71872
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩] } }
def leftRaw : List Term := Proof.Events281.exact71946RawTerms
def rightRaw : List Term := Proof.Events280.exact71923RawTerms
def group : MergeGroup := .operator 71946 71923
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71946) (leftOrdinal := 1)
    (rightResult := 71923) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47518⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71951

namespace LeftMerge71953
def owner : Owner := ⟨.program ⟨257⟩, ⟨47519⟩⟩
def mergeEvent : Nat := 71953
def frameStart : Nat := 71872
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45524⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46683⟩⟩] } }
def rhsRaw : List Term := Proof.Events280.exact71920RawTerms
def group : MergeGroup := .relation 71952
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 71952) (rhsResult := 71920)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47518⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47518⟩⟩) ⟨46683⟩ 71920) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46683⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45524⟩⟩], [⟨.program ⟨257⟩, ⟨46683⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge71953

namespace LeftMerge71961
def owner : Owner := ⟨.program ⟨257⟩, ⟨45772⟩⟩
def mergeEvent : Nat := 71961
def frameStart : Nat := 71872
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45770⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events280.exact71934RawTerms
def rightRaw : List Term := Proof.Events281.exact71957RawTerms
def group : MergeGroup := .operator 71934 71957
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 71934) (leftOrdinal := 0)
    (rightResult := 71957) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45770⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45770⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge71961

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
