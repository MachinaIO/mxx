import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge246910
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246910
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48337⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events963.exact246724RawTerms
def group : MergeGroup := .relation 246909
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 246909) (rhsResult := 246724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 246724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48337⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246910

namespace LeftMerge246911
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246911
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45657⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact246886RawTerms
def rightRaw : List Term := Proof.Events963.exact246727RawTerms
def group : MergeGroup := .operator 246886 246727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 246886) (leftOrdinal := 28)
    (rightResult := 246727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45657⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246911

namespace LeftMerge246913
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246913
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45657⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events963.exact246724RawTerms
def group : MergeGroup := .relation 246912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 246912) (rhsResult := 246724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 246724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246913

namespace LeftMerge246914
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246914
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42973⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact246886RawTerms
def rightRaw : List Term := Proof.Events963.exact246727RawTerms
def group : MergeGroup := .operator 246886 246727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 246886) (leftOrdinal := 27)
    (rightResult := 246727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42973⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246914

namespace LeftMerge246916
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246916
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42973⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events963.exact246724RawTerms
def group : MergeGroup := .relation 246915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 246915) (rhsResult := 246724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 246724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42973⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246916

namespace LeftMerge246917
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246917
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact246886RawTerms
def rightRaw : List Term := Proof.Events963.exact246727RawTerms
def group : MergeGroup := .operator 246886 246727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 246886) (leftOrdinal := 26)
    (rightResult := 246727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246917

namespace LeftMerge246919
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246919
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40293⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events963.exact246724RawTerms
def group : MergeGroup := .relation 246918
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 246918) (rhsResult := 246724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40293⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 246724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40293⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246919

namespace LeftMerge246920
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246920
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact246886RawTerms
def rightRaw : List Term := Proof.Events963.exact246727RawTerms
def group : MergeGroup := .operator 246886 246727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 246886) (leftOrdinal := 25)
    (rightResult := 246727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37617⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246920

namespace LeftMerge246922
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246922
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events963.exact246724RawTerms
def group : MergeGroup := .relation 246921
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 246921) (rhsResult := 246724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37617⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 246724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37617⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246922

namespace LeftMerge246923
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246923
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34937⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact246886RawTerms
def rightRaw : List Term := Proof.Events963.exact246727RawTerms
def group : MergeGroup := .operator 246886 246727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 246886) (leftOrdinal := 24)
    (rightResult := 246727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34937⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246923

namespace LeftMerge246925
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246925
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34937⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events963.exact246724RawTerms
def group : MergeGroup := .relation 246924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 246924) (rhsResult := 246724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34937⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 246724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34937⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246925

namespace LeftMerge246926
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246926
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29273⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact246886RawTerms
def rightRaw : List Term := Proof.Events963.exact246727RawTerms
def group : MergeGroup := .operator 246886 246727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 246886) (leftOrdinal := 22)
    (rightResult := 246727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29273⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246926

namespace LeftMerge246928
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246928
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29273⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events963.exact246724RawTerms
def group : MergeGroup := .relation 246927
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 246927) (rhsResult := 246724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29273⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 246724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29273⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246928

namespace LeftMerge246929
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246929
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact246886RawTerms
def rightRaw : List Term := Proof.Events963.exact246727RawTerms
def group : MergeGroup := .operator 246886 246727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 246886) (leftOrdinal := 21)
    (rightResult := 246727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246929

namespace LeftMerge246931
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246931
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26593⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }
def rhsRaw : List Term := Proof.Events963.exact246724RawTerms
def group : MergeGroup := .relation 246930
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 246930) (rhsResult := 246724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26593⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71172⟩⟩) ⟨68818⟩ 246724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68818⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26593⟩⟩], [⟨.program ⟨257⟩, ⟨68818⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246931

namespace LeftMerge246932
def owner : Owner := ⟨.program ⟨257⟩, ⟨71173⟩⟩
def mergeEvent : Nat := 246932
def frameStart : Nat := 246211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66461⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩] } }
def leftRaw : List Term := Proof.Events964.exact246886RawTerms
def rightRaw : List Term := Proof.Events963.exact246727RawTerms
def group : MergeGroup := .operator 246886 246727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 246886) (leftOrdinal := 35)
    (rightResult := 246727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66461⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71172⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66461⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71172⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge246932

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
