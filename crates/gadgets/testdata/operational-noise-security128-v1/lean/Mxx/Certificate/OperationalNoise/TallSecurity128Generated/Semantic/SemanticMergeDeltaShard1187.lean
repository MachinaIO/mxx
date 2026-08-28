import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge192960
def owner : Owner := ⟨.program ⟨257⟩, ⟨15115⟩⟩
def mergeEvent : Nat := 192960
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192951RawTerms
def rightRaw : List Term := Proof.Events066.exact17095RawTerms
def group : MergeGroup := .operator 192951 17095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192951) (leftOrdinal := 0)
    (rightResult := 17095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192960

namespace LeftMerge192965
def owner : Owner := ⟨.program ⟨257⟩, ⟨47889⟩⟩
def mergeEvent : Nat := 192965
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192961RawTerms
def rightRaw : List Term := Proof.Events753.exact192931RawTerms
def group : MergeGroup := .operator 192961 192931
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192961) (leftOrdinal := 1)
    (rightResult := 192931) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7285⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩], [⟨.program ⟨257⟩, ⟨7285⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192965

namespace LeftMerge192973
def owner : Owner := ⟨.program ⟨257⟩, ⟨49682⟩⟩
def mergeEvent : Nat := 192973
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192967RawTerms
def rightRaw : List Term := Proof.Events753.exact192898RawTerms
def group : MergeGroup := .operator 192967 192898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192967) (leftOrdinal := 1)
    (rightResult := 192898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49681⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192973

namespace LeftMerge192975
def owner : Owner := ⟨.program ⟨257⟩, ⟨49682⟩⟩
def mergeEvent : Nat := 192975
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49161⟩⟩] } }
def rhsRaw : List Term := Proof.Events753.exact192895RawTerms
def group : MergeGroup := .relation 192974
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 192974) (rhsResult := 192895)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49681⟩⟩) ⟨49161⟩ 192895) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49161⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge192975

namespace LeftMerge192976
def owner : Owner := ⟨.program ⟨257⟩, ⟨49682⟩⟩
def mergeEvent : Nat := 192976
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192967RawTerms
def rightRaw : List Term := Proof.Events753.exact192898RawTerms
def group : MergeGroup := .operator 192967 192898
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192967) (leftOrdinal := 0)
    (rightResult := 192898) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49681⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192976

namespace LeftMerge192988
def owner : Owner := ⟨.program ⟨257⟩, ⟨5908⟩⟩
def mergeEvent : Nat := 192988
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192773RawTerms
def rightRaw : List Term := Proof.Events067.exact17158RawTerms
def group : MergeGroup := .operator 192773 17158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192773) (leftOrdinal := 0)
    (rightResult := 17158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge192988

namespace LeftMerge193001
def owner : Owner := ⟨.program ⟨257⟩, ⟨48612⟩⟩
def mergeEvent : Nat := 193001
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events753.exact192984RawTerms
def group : MergeGroup := .operator 192995 192984
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 192984) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48609⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193001

namespace LeftMerge193080
def owner : Owner := ⟨.program ⟨257⟩, ⟨47883⟩⟩
def mergeEvent : Nat := 193080
def frameStart : Nat := 193050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events754.exact193076RawTerms
def rightRaw : List Term := Proof.Events754.exact193073RawTerms
def group : MergeGroup := .operator 193076 193073
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193076) (leftOrdinal := 0)
    (rightResult := 193073) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15111⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193080

namespace LeftMerge193110
def owner : Owner := ⟨.program ⟨257⟩, ⟨49436⟩⟩
def mergeEvent : Nat := 193110
def frameStart : Nat := 193050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events754.exact193106RawTerms
def rightRaw : List Term := Proof.Events754.exact193104RawTerms
def group : MergeGroup := .operator 193106 193104
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193106) (leftOrdinal := 0)
    (rightResult := 193104) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193110

namespace LeftMerge193133
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 193133
def frameStart : Nat := 193050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events754.exact193129RawTerms
def rightRaw : List Term := Proof.Events754.exact193126RawTerms
def group : MergeGroup := .operator 193129 193126
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193129) (leftOrdinal := 0)
    (rightResult := 193126) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193133

namespace LeftMerge193142
def owner : Owner := ⟨.program ⟨257⟩, ⟨49684⟩⟩
def mergeEvent : Nat := 193142
def frameStart : Nat := 193050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩] } }
def leftRaw : List Term := Proof.Events754.exact193138RawTerms
def rightRaw : List Term := Proof.Events754.exact193095RawTerms
def group : MergeGroup := .operator 193138 193095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193138) (leftOrdinal := 0)
    (rightResult := 193095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49681⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193142

namespace LeftMerge193143
def owner : Owner := ⟨.program ⟨257⟩, ⟨49684⟩⟩
def mergeEvent : Nat := 193143
def frameStart : Nat := 193050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩] } }
def leftRaw : List Term := Proof.Events754.exact193138RawTerms
def rightRaw : List Term := Proof.Events754.exact193095RawTerms
def group : MergeGroup := .operator 193138 193095
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193138) (leftOrdinal := 1)
    (rightResult := 193095) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49681⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge193143

namespace LeftMerge193145
def owner : Owner := ⟨.program ⟨257⟩, ⟨49684⟩⟩
def mergeEvent : Nat := 193145
def frameStart : Nat := 193050
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49161⟩⟩] } }
def rhsRaw : List Term := Proof.Events754.exact193092RawTerms
def group : MergeGroup := .relation 193144
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 193144) (rhsResult := 193092)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49681⟩⟩) ⟨49161⟩ 193092) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49161⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15111⟩⟩, ⟨.program ⟨257⟩, ⟨47882⟩⟩], [⟨.program ⟨257⟩, ⟨49161⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge193145

namespace LeftMerge193153
def owner : Owner := ⟨.program ⟨257⟩, ⟨48166⟩⟩
def mergeEvent : Nat := 193153
def frameStart : Nat := 193050
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48164⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events754.exact193106RawTerms
def rightRaw : List Term := Proof.Events754.exact193149RawTerms
def group : MergeGroup := .operator 193106 193149
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193106) (leftOrdinal := 0)
    (rightResult := 193149) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48164⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48164⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193153

namespace LeftMerge193170
def owner : Owner := ⟨.program ⟨257⟩, ⟨48612⟩⟩
def mergeEvent : Nat := 193170
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events754.exact193167RawTerms
def group : MergeGroup := .relation 193169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 193169) (rhsResult := 193167)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 193168 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩) (none) 193167) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193170

namespace LeftMerge193171
def owner : Owner := ⟨.program ⟨257⟩, ⟨48612⟩⟩
def mergeEvent : Nat := 193171
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩] } }
def rhsRaw : List Term := Proof.Events754.exact193167RawTerms
def group : MergeGroup := .relation 193169
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 193169) (rhsResult := 193167)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 193168 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48609⟩⟩]⟩) (none) 193167) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49681⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge193171

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
