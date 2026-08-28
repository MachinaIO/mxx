import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge259872
def owner : Owner := ⟨.program ⟨257⟩, ⟨17306⟩⟩
def mergeEvent : Nat := 259872
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16819⟩⟩] } }
def leftRaw : List Term := Proof.Events1015.exact259868RawTerms
def rightRaw : List Term := Proof.Events1014.exact259682RawTerms
def group : MergeGroup := .operator 259868 259682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259868) (leftOrdinal := 2)
    (rightResult := 259682) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16819⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16819⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨12306⟩⟩, ⟨.program ⟨257⟩, ⟨15354⟩⟩], [⟨.program ⟨257⟩, ⟨16819⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259872

namespace LeftMerge259873
def owner : Owner := ⟨.program ⟨257⟩, ⟨17306⟩⟩
def mergeEvent : Nat := 259873
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩] } }
def leftRaw : List Term := Proof.Events1015.exact259868RawTerms
def rightRaw : List Term := Proof.Events1014.exact259682RawTerms
def group : MergeGroup := .operator 259868 259682
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259868) (leftOrdinal := 1)
    (rightResult := 259682) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7303⟩⟩, ⟨.program ⟨257⟩, ⟨9568⟩⟩, ⟨.program ⟨257⟩, ⟨17304⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259873

namespace LeftMerge259881
def owner : Owner := ⟨.program ⟨257⟩, ⟨17623⟩⟩
def mergeEvent : Nat := 259881
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩] } }
def leftRaw : List Term := Proof.Events1015.exact259875RawTerms
def rightRaw : List Term := Proof.Events1014.exact259598RawTerms
def group : MergeGroup := .operator 259875 259598
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259875) (leftOrdinal := 0)
    (rightResult := 259598) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17621⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259881

namespace LeftMerge259882
def owner : Owner := ⟨.program ⟨257⟩, ⟨17623⟩⟩
def mergeEvent : Nat := 259882
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩] } }
def leftRaw : List Term := Proof.Events1015.exact259875RawTerms
def rightRaw : List Term := Proof.Events1014.exact259598RawTerms
def group : MergeGroup := .operator 259875 259598
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 259875) (leftOrdinal := 1)
    (rightResult := 259598) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17621⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259882

namespace LeftMerge259884
def owner : Owner := ⟨.program ⟨257⟩, ⟨17623⟩⟩
def mergeEvent : Nat := 259884
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16956⟩⟩] } }
def rhsRaw : List Term := Proof.Events1014.exact259595RawTerms
def group : MergeGroup := .relation 259883
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 259883) (rhsResult := 259595)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17621⟩⟩) ⟨16956⟩ 259595) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16956⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge259884

namespace LeftMerge259898
def owner : Owner := ⟨.program ⟨257⟩, ⟨16499⟩⟩
def mergeEvent : Nat := 259898
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events1015.exact259892RawTerms
def group : MergeGroup := .operator 251495 259892
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 259892) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16496⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge259898

namespace LeftMerge260019
def owner : Owner := ⟨.program ⟨257⟩, ⟨17188⟩⟩
def mergeEvent : Nat := 260019
def frameStart : Nat := 259953
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1015.exact260015RawTerms
def rightRaw : List Term := Proof.Events1015.exact260013RawTerms
def group : MergeGroup := .operator 260015 260013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 260015) (leftOrdinal := 0)
    (rightResult := 260013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge260019

namespace LeftMerge260031
def owner : Owner := ⟨.program ⟨257⟩, ⟨17622⟩⟩
def mergeEvent : Nat := 260031
def frameStart : Nat := 259953
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩] } }
def leftRaw : List Term := Proof.Events1015.exact260027RawTerms
def rightRaw : List Term := Proof.Events1015.exact260004RawTerms
def group : MergeGroup := .operator 260027 260004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 260027) (leftOrdinal := 0)
    (rightResult := 260004) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17621⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge260031

namespace LeftMerge260032
def owner : Owner := ⟨.program ⟨257⟩, ⟨17622⟩⟩
def mergeEvent : Nat := 260032
def frameStart : Nat := 259953
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩] } }
def leftRaw : List Term := Proof.Events1015.exact260027RawTerms
def rightRaw : List Term := Proof.Events1015.exact260004RawTerms
def group : MergeGroup := .operator 260027 260004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 260027) (leftOrdinal := 1)
    (rightResult := 260004) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨17621⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge260032

namespace LeftMerge260034
def owner : Owner := ⟨.program ⟨257⟩, ⟨17622⟩⟩
def mergeEvent : Nat := 260034
def frameStart : Nat := 259953
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16956⟩⟩] } }
def rhsRaw : List Term := Proof.Events1015.exact260001RawTerms
def group : MergeGroup := .relation 260033
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 260033) (rhsResult := 260001)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨17621⟩⟩) ⟨16956⟩ 260001) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨16956⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge260034

namespace LeftMerge260042
def owner : Owner := ⟨.program ⟨257⟩, ⟨15956⟩⟩
def mergeEvent : Nat := 260042
def frameStart : Nat := 259953
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1015.exact260015RawTerms
def rightRaw : List Term := Proof.Events1015.exact260038RawTerms
def group : MergeGroup := .operator 260015 260038
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 260015) (leftOrdinal := 0)
    (rightResult := 260038) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge260042

namespace LeftMerge260059
def owner : Owner := ⟨.program ⟨257⟩, ⟨16499⟩⟩
def mergeEvent : Nat := 260059
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }
def rhsRaw : List Term := Proof.Events1015.exact260056RawTerms
def group : MergeGroup := .relation 260058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 260058) (rhsResult := 260056)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 260057 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩) (none) 260056) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7198⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge260059

namespace LeftMerge260060
def owner : Owner := ⟨.program ⟨257⟩, ⟨16499⟩⟩
def mergeEvent : Nat := 260060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩] } }
def rhsRaw : List Term := Proof.Events1015.exact260056RawTerms
def group : MergeGroup := .relation 260058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 260058) (rhsResult := 260056)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 260057 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩) (none) 260056) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge260060

namespace LeftMerge260061
def owner : Owner := ⟨.program ⟨257⟩, ⟨16499⟩⟩
def mergeEvent : Nat := 260061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16956⟩⟩] } }
def rhsRaw : List Term := Proof.Events1015.exact260056RawTerms
def group : MergeGroup := .relation 260058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 260058) (rhsResult := 260056)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 260057 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩) (none) 260056) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15748⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨16956⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15748⟩⟩], [⟨.program ⟨257⟩, ⟨16956⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge260061

namespace LeftMerge260062
def owner : Owner := ⟨.program ⟨257⟩, ⟨16499⟩⟩
def mergeEvent : Nat := 260062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1015.exact260056RawTerms
def group : MergeGroup := .relation 260058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 260058) (rhsResult := 260056)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 260057 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨16496⟩⟩]⟩) (none) 260056) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15955⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨15955⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge260062

namespace LeftMerge260067
def owner : Owner := ⟨.program ⟨257⟩, ⟨17624⟩⟩
def mergeEvent : Nat := 260067
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩] } }
def leftRaw : List Term := Proof.Events1015.exact260063RawTerms
def rightRaw : List Term := Proof.Events1015.exact259885RawTerms
def group : MergeGroup := .operator 260063 259885
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 260063) (leftOrdinal := 0)
    (rightResult := 259885) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7179⟩⟩, ⟨.program ⟨257⟩, ⟨17621⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge260067

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
