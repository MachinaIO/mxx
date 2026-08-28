import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge21861
def owner : Owner := ⟨.program ⟨257⟩, ⟨64345⟩⟩
def mergeEvent : Nat := 21861
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63877⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21857RawTerms
def rightRaw : List Term := Proof.Events084.exact21671RawTerms
def group : MergeGroup := .operator 21857 21671
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21857) (leftOrdinal := 2)
    (rightResult := 21671) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63877⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63877⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨25386⟩⟩, ⟨.program ⟨257⟩, ⟨62231⟩⟩], [⟨.program ⟨257⟩, ⟨63877⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21861

namespace LeftMerge21862
def owner : Owner := ⟨.program ⟨257⟩, ⟨64345⟩⟩
def mergeEvent : Nat := 21862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21857RawTerms
def rightRaw : List Term := Proof.Events084.exact21671RawTerms
def group : MergeGroup := .operator 21857 21671
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21857) (leftOrdinal := 1)
    (rightResult := 21671) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64343⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21862

namespace LeftMerge21870
def owner : Owner := ⟨.program ⟨257⟩, ⟨64604⟩⟩
def mergeEvent : Nat := 21870
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21864RawTerms
def rightRaw : List Term := Proof.Events084.exact21568RawTerms
def group : MergeGroup := .operator 21864 21568
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21864) (leftOrdinal := 1)
    (rightResult := 21568) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64602⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21870

namespace LeftMerge21872
def owner : Owner := ⟨.program ⟨257⟩, ⟨64604⟩⟩
def mergeEvent : Nat := 21872
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64003⟩⟩] } }
def rhsRaw : List Term := Proof.Events084.exact21565RawTerms
def group : MergeGroup := .relation 21871
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 21871) (rhsResult := 21565)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64602⟩⟩) ⟨64003⟩ 21565) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨64003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge21872

namespace LeftMerge21873
def owner : Owner := ⟨.program ⟨257⟩, ⟨64604⟩⟩
def mergeEvent : Nat := 21873
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact21864RawTerms
def rightRaw : List Term := Proof.Events084.exact21568RawTerms
def group : MergeGroup := .operator 21864 21568
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21864) (leftOrdinal := 0)
    (rightResult := 21568) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64602⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21873

namespace LeftMerge21887
def owner : Owner := ⟨.program ⟨257⟩, ⟨63505⟩⟩
def mergeEvent : Nat := 21887
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events085.exact21881RawTerms
def group : MergeGroup := .operator 17169 21881
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 21881) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63502⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge21887

namespace LeftMerge22008
def owner : Owner := ⟨.program ⟨257⟩, ⟨64252⟩⟩
def mergeEvent : Nat := 22008
def frameStart : Nat := 21942
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact22004RawTerms
def rightRaw : List Term := Proof.Events085.exact22002RawTerms
def group : MergeGroup := .operator 22004 22002
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22004) (leftOrdinal := 0)
    (rightResult := 22002) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22008

namespace LeftMerge22020
def owner : Owner := ⟨.program ⟨257⟩, ⟨64603⟩⟩
def mergeEvent : Nat := 22020
def frameStart : Nat := 21942
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22016RawTerms
def rightRaw : List Term := Proof.Events085.exact21993RawTerms
def group : MergeGroup := .operator 22016 21993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22016) (leftOrdinal := 1)
    (rightResult := 21993) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64602⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22020

namespace LeftMerge22022
def owner : Owner := ⟨.program ⟨257⟩, ⟨64603⟩⟩
def mergeEvent : Nat := 22022
def frameStart : Nat := 21942
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64003⟩⟩] } }
def rhsRaw : List Term := Proof.Events085.exact21990RawTerms
def group : MergeGroup := .relation 22021
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22021) (rhsResult := 21990)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64602⟩⟩) ⟨64003⟩ 21990) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨64003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22022

namespace LeftMerge22023
def owner : Owner := ⟨.program ⟨257⟩, ⟨64603⟩⟩
def mergeEvent : Nat := 22023
def frameStart : Nat := 21942
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22016RawTerms
def rightRaw : List Term := Proof.Events085.exact21993RawTerms
def group : MergeGroup := .operator 22016 21993
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22016) (leftOrdinal := 0)
    (rightResult := 21993) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64602⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22023

namespace LeftMerge22031
def owner : Owner := ⟨.program ⟨257⟩, ⟨62917⟩⟩
def mergeEvent : Nat := 22031
def frameStart : Nat := 21942
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62915⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events085.exact22004RawTerms
def rightRaw : List Term := Proof.Events086.exact22027RawTerms
def group : MergeGroup := .operator 22004 22027
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22004) (leftOrdinal := 0)
    (rightResult := 22027) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62915⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22031

namespace LeftMerge22048
def owner : Owner := ⟨.program ⟨257⟩, ⟨63505⟩⟩
def mergeEvent : Nat := 22048
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64003⟩⟩] } }
def rhsRaw : List Term := Proof.Events086.exact22045RawTerms
def group : MergeGroup := .relation 22047
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22047) (rhsResult := 22045)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22046 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩) (none) 22045) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64003⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨64003⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22048

namespace LeftMerge22049
def owner : Owner := ⟨.program ⟨257⟩, ⟨63505⟩⟩
def mergeEvent : Nat := 22049
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩] } }
def rhsRaw : List Term := Proof.Events086.exact22045RawTerms
def group : MergeGroup := .relation 22047
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22047) (rhsResult := 22045)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22046 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩) (none) 22045) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64602⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22049

namespace LeftMerge22050
def owner : Owner := ⟨.program ⟨257⟩, ⟨63505⟩⟩
def mergeEvent : Nat := 22050
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events086.exact22045RawTerms
def group : MergeGroup := .relation 22047
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22047) (rhsResult := 22045)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22046 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩) (none) 22045) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62915⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62915⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22050

namespace LeftMerge22051
def owner : Owner := ⟨.program ⟨257⟩, ⟨63505⟩⟩
def mergeEvent : Nat := 22051
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }
def rhsRaw : List Term := Proof.Events086.exact22045RawTerms
def group : MergeGroup := .relation 22047
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 22047) (rhsResult := 22045)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 22046 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩) (none) 22045) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge22051

namespace LeftMerge22056
def owner : Owner := ⟨.program ⟨257⟩, ⟨64605⟩⟩
def mergeEvent : Nat := 22056
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64003⟩⟩] } }
def leftRaw : List Term := Proof.Events086.exact22052RawTerms
def rightRaw : List Term := Proof.Events085.exact21874RawTerms
def group : MergeGroup := .operator 22052 21874
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 22052) (leftOrdinal := 2)
    (rightResult := 21874) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64003⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64003⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨62738⟩⟩], [⟨.program ⟨257⟩, ⟨64003⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge22056

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
