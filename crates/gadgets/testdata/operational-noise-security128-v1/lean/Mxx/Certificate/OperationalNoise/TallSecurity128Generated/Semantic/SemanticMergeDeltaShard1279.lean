import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge207600
def owner : Owner := ⟨.program ⟨257⟩, ⟨49660⟩⟩
def mergeEvent : Nat := 207600
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }
def rhsRaw : List Term := Proof.Events810.exact207520RawTerms
def group : MergeGroup := .relation 207599
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 207599) (rhsResult := 207520)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49659⟩⟩) ⟨49149⟩ 207520) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨49149⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge207600

namespace LeftMerge207601
def owner : Owner := ⟨.program ⟨257⟩, ⟨49660⟩⟩
def mergeEvent : Nat := 207601
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩] } }
def leftRaw : List Term := Proof.Events810.exact207592RawTerms
def rightRaw : List Term := Proof.Events810.exact207523RawTerms
def group : MergeGroup := .operator 207592 207523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207592) (leftOrdinal := 0)
    (rightResult := 207523) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49659⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207601

namespace LeftMerge207613
def owner : Owner := ⟨.program ⟨257⟩, ⟨5598⟩⟩
def mergeEvent : Nat := 207613
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }
def leftRaw : List Term := Proof.Events810.exact207398RawTerms
def rightRaw : List Term := Proof.Events067.exact17158RawTerms
def group : MergeGroup := .operator 207398 17158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207398) (leftOrdinal := 0)
    (rightResult := 17158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207613

namespace LeftMerge207626
def owner : Owner := ⟨.program ⟨257⟩, ⟨48592⟩⟩
def mergeEvent : Nat := 207626
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events810.exact207609RawTerms
def group : MergeGroup := .operator 207620 207609
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 207609) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48589⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207626

namespace LeftMerge207705
def owner : Owner := ⟨.program ⟨257⟩, ⟨47835⟩⟩
def mergeEvent : Nat := 207705
def frameStart : Nat := 207675
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events811.exact207701RawTerms
def rightRaw : List Term := Proof.Events811.exact207698RawTerms
def group : MergeGroup := .operator 207701 207698
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207701) (leftOrdinal := 0)
    (rightResult := 207698) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207705

namespace LeftMerge207735
def owner : Owner := ⟨.program ⟨257⟩, ⟨49428⟩⟩
def mergeEvent : Nat := 207735
def frameStart : Nat := 207675
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207731RawTerms
def rightRaw : List Term := Proof.Events811.exact207729RawTerms
def group : MergeGroup := .operator 207731 207729
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207731) (leftOrdinal := 0)
    (rightResult := 207729) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207735

namespace LeftMerge207758
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 207758
def frameStart : Nat := 207675
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207754RawTerms
def rightRaw : List Term := Proof.Events811.exact207751RawTerms
def group : MergeGroup := .operator 207754 207751
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207754) (leftOrdinal := 0)
    (rightResult := 207751) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207758

namespace LeftMerge207767
def owner : Owner := ⟨.program ⟨257⟩, ⟨49662⟩⟩
def mergeEvent : Nat := 207767
def frameStart : Nat := 207675
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207763RawTerms
def rightRaw : List Term := Proof.Events811.exact207720RawTerms
def group : MergeGroup := .operator 207763 207720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207763) (leftOrdinal := 0)
    (rightResult := 207720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49659⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207767

namespace LeftMerge207768
def owner : Owner := ⟨.program ⟨257⟩, ⟨49662⟩⟩
def mergeEvent : Nat := 207768
def frameStart : Nat := 207675
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207763RawTerms
def rightRaw : List Term := Proof.Events811.exact207720RawTerms
def group : MergeGroup := .operator 207763 207720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207763) (leftOrdinal := 1)
    (rightResult := 207720) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49659⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge207768

namespace LeftMerge207770
def owner : Owner := ⟨.program ⟨257⟩, ⟨49662⟩⟩
def mergeEvent : Nat := 207770
def frameStart : Nat := 207675
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }
def rhsRaw : List Term := Proof.Events811.exact207717RawTerms
def group : MergeGroup := .relation 207769
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 207769) (rhsResult := 207717)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49659⟩⟩) ⟨49149⟩ 207717) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨49149⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge207770

namespace LeftMerge207778
def owner : Owner := ⟨.program ⟨257⟩, ⟨48150⟩⟩
def mergeEvent : Nat := 207778
def frameStart : Nat := 207675
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207731RawTerms
def rightRaw : List Term := Proof.Events811.exact207774RawTerms
def group : MergeGroup := .operator 207731 207774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207731) (leftOrdinal := 0)
    (rightResult := 207774) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207778

namespace LeftMerge207795
def owner : Owner := ⟨.program ⟨257⟩, ⟨48592⟩⟩
def mergeEvent : Nat := 207795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events811.exact207792RawTerms
def group : MergeGroup := .relation 207794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 207794) (rhsResult := 207792)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 207793 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩) (none) 207792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207795

namespace LeftMerge207796
def owner : Owner := ⟨.program ⟨257⟩, ⟨48592⟩⟩
def mergeEvent : Nat := 207796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩] } }
def rhsRaw : List Term := Proof.Events811.exact207792RawTerms
def group : MergeGroup := .relation 207794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 207794) (rhsResult := 207792)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 207793 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩) (none) 207792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49659⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge207796

namespace LeftMerge207797
def owner : Owner := ⟨.program ⟨257⟩, ⟨48592⟩⟩
def mergeEvent : Nat := 207797
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }
def rhsRaw : List Term := Proof.Events811.exact207792RawTerms
def group : MergeGroup := .relation 207794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 207794) (rhsResult := 207792)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 207793 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩) (none) 207792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨49149⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge207797

namespace LeftMerge207798
def owner : Owner := ⟨.program ⟨257⟩, ⟨48592⟩⟩
def mergeEvent : Nat := 207798
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events811.exact207792RawTerms
def group : MergeGroup := .relation 207794
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 207794) (rhsResult := 207792)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 207793 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48589⟩⟩]⟩) (none) 207792) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48148⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨48148⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge207798

namespace LeftMerge207803
def owner : Owner := ⟨.program ⟨257⟩, ⟨49661⟩⟩
def mergeEvent : Nat := 207803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207799RawTerms
def rightRaw : List Term := Proof.Events810.exact207602RawTerms
def group : MergeGroup := .operator 207799 207602
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207799) (leftOrdinal := 2)
    (rightResult := 207602) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49149⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨15081⟩⟩, ⟨.program ⟨257⟩, ⟨47834⟩⟩], [⟨.program ⟨257⟩, ⟨49149⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge207803

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
