import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge197028
def owner : Owner := ⟨.program ⟨257⟩, ⟨67793⟩⟩
def mergeEvent : Nat := 197028
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68542⟩⟩] } }
def rhsRaw : List Term := Proof.Events769.exact197023RawTerms
def group : MergeGroup := .relation 197025
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197025) (rhsResult := 197023)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67790⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197024 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67790⟩⟩]⟩) (none) 197023) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68542⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨68542⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197028

namespace LeftMerge197029
def owner : Owner := ⟨.program ⟨257⟩, ⟨67793⟩⟩
def mergeEvent : Nat := 197029
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events769.exact197023RawTerms
def group : MergeGroup := .relation 197025
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197025) (rhsResult := 197023)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67790⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197024 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67790⟩⟩]⟩) (none) 197023) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197029

namespace LeftMerge197034
def owner : Owner := ⟨.program ⟨257⟩, ⟨69264⟩⟩
def mergeEvent : Nat := 197034
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68542⟩⟩] } }
def leftRaw : List Term := Proof.Events769.exact197030RawTerms
def rightRaw : List Term := Proof.Events768.exact196844RawTerms
def group : MergeGroup := .operator 197030 196844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197030) (leftOrdinal := 2)
    (rightResult := 196844) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68542⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68542⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨25754⟩⟩, ⟨.program ⟨257⟩, ⟨65499⟩⟩], [⟨.program ⟨257⟩, ⟨68542⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197034

namespace LeftMerge197035
def owner : Owner := ⟨.program ⟨257⟩, ⟨69264⟩⟩
def mergeEvent : Nat := 197035
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69262⟩⟩] } }
def leftRaw : List Term := Proof.Events769.exact197030RawTerms
def rightRaw : List Term := Proof.Events768.exact196844RawTerms
def group : MergeGroup := .operator 197030 196844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197030) (leftOrdinal := 1)
    (rightResult := 196844) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69262⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69262⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7294⟩⟩, ⟨.program ⟨257⟩, ⟨9541⟩⟩, ⟨.program ⟨257⟩, ⟨69262⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197035

namespace LeftMerge197043
def owner : Owner := ⟨.program ⟨257⟩, ⟨70337⟩⟩
def mergeEvent : Nat := 197043
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩] } }
def leftRaw : List Term := Proof.Events769.exact197037RawTerms
def rightRaw : List Term := Proof.Events768.exact196760RawTerms
def group : MergeGroup := .operator 197037 196760
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197037) (leftOrdinal := 0)
    (rightResult := 196760) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70335⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197043

namespace LeftMerge197044
def owner : Owner := ⟨.program ⟨257⟩, ⟨70337⟩⟩
def mergeEvent : Nat := 197044
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩] } }
def leftRaw : List Term := Proof.Events769.exact197037RawTerms
def rightRaw : List Term := Proof.Events768.exact196760RawTerms
def group : MergeGroup := .operator 197037 196760
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197037) (leftOrdinal := 1)
    (rightResult := 196760) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70335⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197044

namespace LeftMerge197046
def owner : Owner := ⟨.program ⟨257⟩, ⟨70337⟩⟩
def mergeEvent : Nat := 197046
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68700⟩⟩] } }
def rhsRaw : List Term := Proof.Events768.exact196757RawTerms
def group : MergeGroup := .relation 197045
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197045) (rhsResult := 196757)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70335⟩⟩) ⟨68700⟩ 196757) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68700⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨68700⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197046

namespace LeftMerge197060
def owner : Owner := ⟨.program ⟨257⟩, ⟨68120⟩⟩
def mergeEvent : Nat := 197060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68117⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events769.exact197054RawTerms
def group : MergeGroup := .operator 192995 197054
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 197054) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68117⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68117⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197060

namespace LeftMerge197181
def owner : Owner := ⟨.program ⟨257⟩, ⟨69017⟩⟩
def mergeEvent : Nat := 197181
def frameStart : Nat := 197115
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events770.exact197177RawTerms
def rightRaw : List Term := Proof.Events770.exact197175RawTerms
def group : MergeGroup := .operator 197177 197175
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197177) (leftOrdinal := 0)
    (rightResult := 197175) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197181

namespace LeftMerge197193
def owner : Owner := ⟨.program ⟨257⟩, ⟨70336⟩⟩
def mergeEvent : Nat := 197193
def frameStart : Nat := 197115
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩] } }
def leftRaw : List Term := Proof.Events770.exact197189RawTerms
def rightRaw : List Term := Proof.Events770.exact197166RawTerms
def group : MergeGroup := .operator 197189 197166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197189) (leftOrdinal := 0)
    (rightResult := 197166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70335⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197193

namespace LeftMerge197194
def owner : Owner := ⟨.program ⟨257⟩, ⟨70336⟩⟩
def mergeEvent : Nat := 197194
def frameStart : Nat := 197115
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩] } }
def leftRaw : List Term := Proof.Events770.exact197189RawTerms
def rightRaw : List Term := Proof.Events770.exact197166RawTerms
def group : MergeGroup := .operator 197189 197166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197189) (leftOrdinal := 1)
    (rightResult := 197166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70335⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197194

namespace LeftMerge197196
def owner : Owner := ⟨.program ⟨257⟩, ⟨70336⟩⟩
def mergeEvent : Nat := 197196
def frameStart : Nat := 197115
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68700⟩⟩] } }
def rhsRaw : List Term := Proof.Events770.exact197163RawTerms
def group : MergeGroup := .relation 197195
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197195) (rhsResult := 197163)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70335⟩⟩) ⟨68700⟩ 197163) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68700⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨68700⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197196

namespace LeftMerge197204
def owner : Owner := ⟨.program ⟨257⟩, ⟨66752⟩⟩
def mergeEvent : Nat := 197204
def frameStart : Nat := 197115
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66741⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events770.exact197177RawTerms
def rightRaw : List Term := Proof.Events770.exact197200RawTerms
def group : MergeGroup := .operator 197177 197200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 197177) (leftOrdinal := 0)
    (rightResult := 197200) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66741⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66741⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197204

namespace LeftMerge197221
def owner : Owner := ⟨.program ⟨257⟩, ⟨68120⟩⟩
def mergeEvent : Nat := 197221
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }
def rhsRaw : List Term := Proof.Events770.exact197218RawTerms
def group : MergeGroup := .relation 197220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197220) (rhsResult := 197218)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68117⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197219 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68117⟩⟩]⟩) (none) 197218) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197221

namespace LeftMerge197222
def owner : Owner := ⟨.program ⟨257⟩, ⟨68120⟩⟩
def mergeEvent : Nat := 197222
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩] } }
def rhsRaw : List Term := Proof.Events770.exact197218RawTerms
def group : MergeGroup := .relation 197220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197220) (rhsResult := 197218)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68117⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197219 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68117⟩⟩]⟩) (none) 197218) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨70335⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge197222

namespace LeftMerge197223
def owner : Owner := ⟨.program ⟨257⟩, ⟨68120⟩⟩
def mergeEvent : Nat := 197223
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68700⟩⟩] } }
def rhsRaw : List Term := Proof.Events770.exact197218RawTerms
def group : MergeGroup := .relation 197220
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 197220) (rhsResult := 197218)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68117⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 197219 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68117⟩⟩]⟩) (none) 197218) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68700⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨65804⟩⟩], [⟨.program ⟨257⟩, ⟨68700⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge197223

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
