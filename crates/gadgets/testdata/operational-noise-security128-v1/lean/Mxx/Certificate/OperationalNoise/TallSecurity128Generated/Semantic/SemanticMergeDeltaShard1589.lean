import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge257908
def owner : Owner := ⟨.program ⟨257⟩, ⟨52467⟩⟩
def mergeEvent : Nat := 257908
def frameStart : Nat := 257816
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩] } }
def leftRaw : List Term := Proof.Events1007.exact257904RawTerms
def rightRaw : List Term := Proof.Events1007.exact257861RawTerms
def group : MergeGroup := .operator 257904 257861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257904) (leftOrdinal := 0)
    (rightResult := 257861) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52464⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257908

namespace LeftMerge257909
def owner : Owner := ⟨.program ⟨257⟩, ⟨52467⟩⟩
def mergeEvent : Nat := 257909
def frameStart : Nat := 257816
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩] } }
def leftRaw : List Term := Proof.Events1007.exact257904RawTerms
def rightRaw : List Term := Proof.Events1007.exact257861RawTerms
def group : MergeGroup := .operator 257904 257861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257904) (leftOrdinal := 1)
    (rightResult := 257861) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52464⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257909

namespace LeftMerge257911
def owner : Owner := ⟨.program ⟨257⟩, ⟨52467⟩⟩
def mergeEvent : Nat := 257911
def frameStart : Nat := 257816
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51979⟩⟩] } }
def rhsRaw : List Term := Proof.Events1007.exact257858RawTerms
def group : MergeGroup := .relation 257910
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257910) (rhsResult := 257858)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52464⟩⟩) ⟨51979⟩ 257858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51979⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], [⟨.program ⟨257⟩, ⟨51979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257911

namespace LeftMerge257919
def owner : Owner := ⟨.program ⟨257⟩, ⟨50850⟩⟩
def mergeEvent : Nat := 257919
def frameStart : Nat := 257816
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1007.exact257872RawTerms
def rightRaw : List Term := Proof.Events1007.exact257915RawTerms
def group : MergeGroup := .operator 257872 257915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257872) (leftOrdinal := 0)
    (rightResult := 257915) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50848⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257919

namespace LeftMerge257936
def owner : Owner := ⟨.program ⟨257⟩, ⟨51402⟩⟩
def mergeEvent : Nat := 257936
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events1007.exact257933RawTerms
def group : MergeGroup := .relation 257935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257935) (rhsResult := 257933)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 257934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51399⟩⟩]⟩) (none) 257933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257936

namespace LeftMerge257937
def owner : Owner := ⟨.program ⟨257⟩, ⟨51402⟩⟩
def mergeEvent : Nat := 257937
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩] } }
def rhsRaw : List Term := Proof.Events1007.exact257933RawTerms
def group : MergeGroup := .relation 257935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257935) (rhsResult := 257933)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 257934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51399⟩⟩]⟩) (none) 257933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257937

namespace LeftMerge257938
def owner : Owner := ⟨.program ⟨257⟩, ⟨51402⟩⟩
def mergeEvent : Nat := 257938
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51979⟩⟩] } }
def rhsRaw : List Term := Proof.Events1007.exact257933RawTerms
def group : MergeGroup := .relation 257935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257935) (rhsResult := 257933)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 257934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51399⟩⟩]⟩) (none) 257933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51979⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], [⟨.program ⟨257⟩, ⟨51979⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257938

namespace LeftMerge257939
def owner : Owner := ⟨.program ⟨257⟩, ⟨51402⟩⟩
def mergeEvent : Nat := 257939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨50848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1007.exact257933RawTerms
def group : MergeGroup := .relation 257935
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257935) (rhsResult := 257933)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51399⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 257934 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51399⟩⟩]⟩) (none) 257933) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨50848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257939

namespace LeftMerge257944
def owner : Owner := ⟨.program ⟨257⟩, ⟨52466⟩⟩
def mergeEvent : Nat := 257944
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51979⟩⟩] } }
def leftRaw : List Term := Proof.Events1007.exact257940RawTerms
def rightRaw : List Term := Proof.Events1006.exact257754RawTerms
def group : MergeGroup := .operator 257940 257754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257940) (leftOrdinal := 2)
    (rightResult := 257754) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51979⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨51979⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨24470⟩⟩, ⟨.program ⟨257⟩, ⟨50410⟩⟩], [⟨.program ⟨257⟩, ⟨51979⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257944

namespace LeftMerge257945
def owner : Owner := ⟨.program ⟨257⟩, ⟨52466⟩⟩
def mergeEvent : Nat := 257945
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩] } }
def leftRaw : List Term := Proof.Events1007.exact257940RawTerms
def rightRaw : List Term := Proof.Events1006.exact257754RawTerms
def group : MergeGroup := .operator 257940 257754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257940) (leftOrdinal := 1)
    (rightResult := 257754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52464⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257945

namespace LeftMerge257953
def owner : Owner := ⟨.program ⟨257⟩, ⟨52799⟩⟩
def mergeEvent : Nat := 257953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52797⟩⟩] } }
def leftRaw : List Term := Proof.Events1007.exact257947RawTerms
def rightRaw : List Term := Proof.Events1006.exact257670RawTerms
def group : MergeGroup := .operator 257947 257670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257947) (leftOrdinal := 0)
    (rightResult := 257670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52797⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52797⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257953

namespace LeftMerge257954
def owner : Owner := ⟨.program ⟨257⟩, ⟨52799⟩⟩
def mergeEvent : Nat := 257954
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨50848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52797⟩⟩] } }
def leftRaw : List Term := Proof.Events1007.exact257947RawTerms
def rightRaw : List Term := Proof.Events1006.exact257670RawTerms
def group : MergeGroup := .operator 257947 257670
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 257947) (leftOrdinal := 1)
    (rightResult := 257670) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨50848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52797⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨50848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52797⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257954

namespace LeftMerge257956
def owner : Owner := ⟨.program ⟨257⟩, ⟨52799⟩⟩
def mergeEvent : Nat := 257956
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨50848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52116⟩⟩] } }
def rhsRaw : List Term := Proof.Events1006.exact257667RawTerms
def group : MergeGroup := .relation 257955
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 257955) (rhsResult := 257667)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨50848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52797⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52797⟩⟩) ⟨52116⟩ 257667) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52116⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨50848⟩⟩], [⟨.program ⟨257⟩, ⟨52116⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge257956

namespace LeftMerge257970
def owner : Owner := ⟨.program ⟨257⟩, ⟨51659⟩⟩
def mergeEvent : Nat := 257970
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51656⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1007.exact257964RawTerms
def group : MergeGroup := .operator 251495 257964
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 257964) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51656⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51656⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge257970

namespace LeftMerge258091
def owner : Owner := ⟨.program ⟨257⟩, ⟨52348⟩⟩
def mergeEvent : Nat := 258091
def frameStart : Nat := 258025
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1008.exact258087RawTerms
def rightRaw : List Term := Proof.Events1008.exact258085RawTerms
def group : MergeGroup := .operator 258087 258085
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258087) (leftOrdinal := 0)
    (rightResult := 258085) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50848⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258091

namespace LeftMerge258103
def owner : Owner := ⟨.program ⟨257⟩, ⟨52798⟩⟩
def mergeEvent : Nat := 258103
def frameStart : Nat := 258025
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52797⟩⟩] } }
def leftRaw : List Term := Proof.Events1008.exact258099RawTerms
def rightRaw : List Term := Proof.Events1008.exact258076RawTerms
def group : MergeGroup := .operator 258099 258076
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 258099) (leftOrdinal := 0)
    (rightResult := 258076) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52797⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52797⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge258103

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
