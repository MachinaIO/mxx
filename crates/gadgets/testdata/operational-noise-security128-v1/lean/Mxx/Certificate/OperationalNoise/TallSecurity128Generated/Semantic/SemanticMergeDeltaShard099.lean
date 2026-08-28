import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge19821
def owner : Owner := ⟨.program ⟨257⟩, ⟨36166⟩⟩
def mergeEvent : Nat := 19821
def frameStart : Nat := 19729
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19817RawTerms
def rightRaw : List Term := Proof.Events077.exact19774RawTerms
def group : MergeGroup := .operator 19817 19774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19817) (leftOrdinal := 1)
    (rightResult := 19774) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19821

namespace LeftMerge19823
def owner : Owner := ⟨.program ⟨257⟩, ⟨36166⟩⟩
def mergeEvent : Nat := 19823
def frameStart : Nat := 19729
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35697⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19771RawTerms
def group : MergeGroup := .relation 19822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19822) (rhsResult := 19771)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36163⟩⟩) ⟨35697⟩ 19771) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35697⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], [⟨.program ⟨257⟩, ⟨35697⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19823

namespace LeftMerge19824
def owner : Owner := ⟨.program ⟨257⟩, ⟨36166⟩⟩
def mergeEvent : Nat := 19824
def frameStart : Nat := 19729
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19817RawTerms
def rightRaw : List Term := Proof.Events077.exact19774RawTerms
def group : MergeGroup := .operator 19817 19774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19817) (leftOrdinal := 0)
    (rightResult := 19774) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36163⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19824

namespace LeftMerge19832
def owner : Owner := ⟨.program ⟨257⟩, ⟨34680⟩⟩
def mergeEvent : Nat := 19832
def frameStart : Nat := 19729
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19785RawTerms
def rightRaw : List Term := Proof.Events077.exact19828RawTerms
def group : MergeGroup := .operator 19785 19828
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19785) (leftOrdinal := 0)
    (rightResult := 19828) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19832

namespace LeftMerge19849
def owner : Owner := ⟨.program ⟨257⟩, ⟨35105⟩⟩
def mergeEvent : Nat := 19849
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35697⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19846RawTerms
def group : MergeGroup := .relation 19848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19848) (rhsResult := 19846)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35102⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19847 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35102⟩⟩]⟩) (none) 19846) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35697⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], [⟨.program ⟨257⟩, ⟨35697⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19849

namespace LeftMerge19850
def owner : Owner := ⟨.program ⟨257⟩, ⟨35105⟩⟩
def mergeEvent : Nat := 19850
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19846RawTerms
def group : MergeGroup := .relation 19848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19848) (rhsResult := 19846)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35102⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19847 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35102⟩⟩]⟩) (none) 19846) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19850

namespace LeftMerge19851
def owner : Owner := ⟨.program ⟨257⟩, ⟨35105⟩⟩
def mergeEvent : Nat := 19851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19846RawTerms
def group : MergeGroup := .relation 19848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19848) (rhsResult := 19846)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35102⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19847 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35102⟩⟩]⟩) (none) 19846) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19851

namespace LeftMerge19852
def owner : Owner := ⟨.program ⟨257⟩, ⟨35105⟩⟩
def mergeEvent : Nat := 19852
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events077.exact19846RawTerms
def group : MergeGroup := .relation 19848
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19848) (rhsResult := 19846)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35102⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 19847 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35102⟩⟩]⟩) (none) 19846) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19852

namespace LeftMerge19857
def owner : Owner := ⟨.program ⟨257⟩, ⟨36165⟩⟩
def mergeEvent : Nat := 19857
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35697⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19853RawTerms
def rightRaw : List Term := Proof.Events076.exact19667RawTerms
def group : MergeGroup := .operator 19853 19667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19853) (leftOrdinal := 2)
    (rightResult := 19667) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35697⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35697⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨13451⟩⟩, ⟨.program ⟨257⟩, ⟨34226⟩⟩], [⟨.program ⟨257⟩, ⟨35697⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19857

namespace LeftMerge19858
def owner : Owner := ⟨.program ⟨257⟩, ⟨36165⟩⟩
def mergeEvent : Nat := 19858
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19853RawTerms
def rightRaw : List Term := Proof.Events076.exact19667RawTerms
def group : MergeGroup := .operator 19853 19667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19853) (leftOrdinal := 1)
    (rightResult := 19667) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19858

namespace LeftMerge19866
def owner : Owner := ⟨.program ⟨257⟩, ⟨36413⟩⟩
def mergeEvent : Nat := 19866
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36411⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19860RawTerms
def rightRaw : List Term := Proof.Events076.exact19564RawTerms
def group : MergeGroup := .operator 19860 19564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19860) (leftOrdinal := 1)
    (rightResult := 19564) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36411⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36411⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19866

namespace LeftMerge19868
def owner : Owner := ⟨.program ⟨257⟩, ⟨36413⟩⟩
def mergeEvent : Nat := 19868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35823⟩⟩] } }
def rhsRaw : List Term := Proof.Events076.exact19561RawTerms
def group : MergeGroup := .relation 19867
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 19867) (rhsResult := 19561)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36411⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36411⟩⟩) ⟨35823⟩ 19561) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35823⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨35823⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge19868

namespace LeftMerge19869
def owner : Owner := ⟨.program ⟨257⟩, ⟨36413⟩⟩
def mergeEvent : Nat := 19869
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36411⟩⟩] } }
def leftRaw : List Term := Proof.Events077.exact19860RawTerms
def rightRaw : List Term := Proof.Events076.exact19564RawTerms
def group : MergeGroup := .operator 19860 19564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 19860) (leftOrdinal := 0)
    (rightResult := 19564) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36411⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36411⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19869

namespace LeftMerge19883
def owner : Owner := ⟨.program ⟨257⟩, ⟨35325⟩⟩
def mergeEvent : Nat := 19883
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35322⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events077.exact19877RawTerms
def group : MergeGroup := .operator 17169 19877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 19877) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35322⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35322⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge19883

namespace LeftMerge20004
def owner : Owner := ⟨.program ⟨257⟩, ⟨36072⟩⟩
def mergeEvent : Nat := 20004
def frameStart : Nat := 19938
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events078.exact20000RawTerms
def rightRaw : List Term := Proof.Events078.exact19998RawTerms
def group : MergeGroup := .operator 20000 19998
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20000) (leftOrdinal := 0)
    (rightResult := 19998) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20004

namespace LeftMerge20016
def owner : Owner := ⟨.program ⟨257⟩, ⟨36412⟩⟩
def mergeEvent : Nat := 20016
def frameStart : Nat := 19938
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36411⟩⟩] } }
def leftRaw : List Term := Proof.Events078.exact20012RawTerms
def rightRaw : List Term := Proof.Events078.exact19989RawTerms
def group : MergeGroup := .operator 20012 19989
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20012) (leftOrdinal := 1)
    (rightResult := 19989) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34678⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36411⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34678⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36411⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge20016

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
