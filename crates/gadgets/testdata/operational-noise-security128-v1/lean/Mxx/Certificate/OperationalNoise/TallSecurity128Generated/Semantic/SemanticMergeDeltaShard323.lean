import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge56822
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56822
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51313⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 30)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51313⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56822

namespace LeftMerge56824
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56824
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51313⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56823) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨51313⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56824

namespace LeftMerge56825
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56825
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 23)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56825

namespace LeftMerge56827
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56827
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32258⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56826) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨32258⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56827

namespace LeftMerge56828
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56828
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 20)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56828

namespace LeftMerge56830
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56830
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22238⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56829
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56829) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨22238⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56830

namespace LeftMerge56831
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56831
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 19)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56831

namespace LeftMerge56833
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56833
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56832
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56832) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56833

namespace LeftMerge56834
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56834
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56761RawTerms
def rightRaw : List Term := Proof.Events221.exact56602RawTerms
def group : MergeGroup := .operator 56761 56602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56761) (leftOrdinal := 18)
    (rightResult := 56602) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨16163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71501⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56834

namespace LeftMerge56836
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def mergeEvent : Nat := 56836
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨16163⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }
def rhsRaw : List Term := Proof.Events221.exact56599RawTerms
def group : MergeGroup := .relation 56835
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56835) (rhsResult := 56599)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71501⟩⟩) ⟨68878⟩ 56599) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68878⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨16163⟩⟩], [⟨.program ⟨257⟩, ⟨68878⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56836

namespace LeftMerge56844
def owner : Owner := ⟨.program ⟨257⟩, ⟨67628⟩⟩
def mergeEvent : Nat := 56844
def frameStart : Nat := 56086
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨67626⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events221.exact56613RawTerms
def rightRaw : List Term := Proof.Events222.exact56840RawTerms
def group : MergeGroup := .operator 56613 56840
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 56613) (leftOrdinal := 0)
    (rightResult := 56840) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67626⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨67626⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56844

namespace LeftMerge56861
def owner : Owner := ⟨.program ⟨257⟩, ⟨68453⟩⟩
def mergeEvent : Nat := 56861
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }
def rhsRaw : List Term := Proof.Events222.exact56858RawTerms
def group : MergeGroup := .relation 56860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56860) (rhsResult := 56858)
    (sourceTermOrdinal := 18) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 56859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩) (none) 56858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7233⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7233⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge56861

namespace LeftMerge56862
def owner : Owner := ⟨.program ⟨257⟩, ⟨68453⟩⟩
def mergeEvent : Nat := 56862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def rhsRaw : List Term := Proof.Events222.exact56858RawTerms
def group : MergeGroup := .relation 56860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56860) (rhsResult := 56858)
    (sourceTermOrdinal := 17) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 56859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩) (none) 56858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56862

namespace LeftMerge56863
def owner : Owner := ⟨.program ⟨257⟩, ⟨68453⟩⟩
def mergeEvent : Nat := 56863
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def rhsRaw : List Term := Proof.Events222.exact56858RawTerms
def group : MergeGroup := .relation 56860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56860) (rhsResult := 56858)
    (sourceTermOrdinal := 16) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 56859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩) (none) 56858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56863

namespace LeftMerge56864
def owner : Owner := ⟨.program ⟨257⟩, ⟨68453⟩⟩
def mergeEvent : Nat := 56864
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def rhsRaw : List Term := Proof.Events222.exact56858RawTerms
def group : MergeGroup := .relation 56860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56860) (rhsResult := 56858)
    (sourceTermOrdinal := 15) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 56859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩) (none) 56858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56864

namespace LeftMerge56865
def owner : Owner := ⟨.program ⟨257⟩, ⟨68453⟩⟩
def mergeEvent : Nat := 56865
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }
def rhsRaw : List Term := Proof.Events222.exact56858RawTerms
def group : MergeGroup := .relation 56860
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 56860) (rhsResult := 56858)
    (sourceTermOrdinal := 14) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 56859 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68450⟩⟩]⟩) (none) 56858) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨71501⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge56865

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
