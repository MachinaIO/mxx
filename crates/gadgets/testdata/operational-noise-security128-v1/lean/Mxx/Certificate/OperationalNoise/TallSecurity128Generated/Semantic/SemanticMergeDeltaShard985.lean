import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge162032
def owner : Owner := ⟨.program ⟨257⟩, ⟨55833⟩⟩
def mergeEvent : Nat := 162032
def frameStart : Nat := 161954
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩] } }
def leftRaw : List Term := Proof.Events632.exact162028RawTerms
def rightRaw : List Term := Proof.Events632.exact162005RawTerms
def group : MergeGroup := .operator 162028 162005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162028) (leftOrdinal := 0)
    (rightResult := 162005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55832⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162032

namespace LeftMerge162033
def owner : Owner := ⟨.program ⟨257⟩, ⟨55833⟩⟩
def mergeEvent : Nat := 162033
def frameStart : Nat := 161954
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩] } }
def leftRaw : List Term := Proof.Events632.exact162028RawTerms
def rightRaw : List Term := Proof.Events632.exact162005RawTerms
def group : MergeGroup := .operator 162028 162005
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162028) (leftOrdinal := 1)
    (rightResult := 162005) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55832⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162033

namespace LeftMerge162035
def owner : Owner := ⟨.program ⟨257⟩, ⟨55833⟩⟩
def mergeEvent : Nat := 162035
def frameStart : Nat := 161954
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55113⟩⟩] } }
def rhsRaw : List Term := Proof.Events632.exact162002RawTerms
def group : MergeGroup := .relation 162034
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162034) (rhsResult := 162002)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨53844⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55832⟩⟩) ⟨55113⟩ 162002) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨53844⟩⟩], [⟨.program ⟨257⟩, ⟨55113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162035

namespace LeftMerge162043
def owner : Owner := ⟨.program ⟨257⟩, ⟨54091⟩⟩
def mergeEvent : Nat := 162043
def frameStart : Nat := 161954
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events632.exact162016RawTerms
def rightRaw : List Term := Proof.Events632.exact162039RawTerms
def group : MergeGroup := .operator 162016 162039
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162016) (leftOrdinal := 0)
    (rightResult := 162039) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162043

namespace LeftMerge162060
def owner : Owner := ⟨.program ⟨257⟩, ⟨54675⟩⟩
def mergeEvent : Nat := 162060
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }
def rhsRaw : List Term := Proof.Events633.exact162057RawTerms
def group : MergeGroup := .relation 162059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162059) (rhsResult := 162057)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54672⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 162058 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54672⟩⟩]⟩) (none) 162057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162060

namespace LeftMerge162061
def owner : Owner := ⟨.program ⟨257⟩, ⟨54675⟩⟩
def mergeEvent : Nat := 162061
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩] } }
def rhsRaw : List Term := Proof.Events633.exact162057RawTerms
def group : MergeGroup := .relation 162059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162059) (rhsResult := 162057)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54672⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 162058 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54672⟩⟩]⟩) (none) 162057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162061

namespace LeftMerge162062
def owner : Owner := ⟨.program ⟨257⟩, ⟨54675⟩⟩
def mergeEvent : Nat := 162062
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨53844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55113⟩⟩] } }
def rhsRaw : List Term := Proof.Events633.exact162057RawTerms
def group : MergeGroup := .relation 162059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162059) (rhsResult := 162057)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54672⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 162058 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54672⟩⟩]⟩) (none) 162057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55113⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨53844⟩⟩], [⟨.program ⟨257⟩, ⟨55113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162062

namespace LeftMerge162063
def owner : Owner := ⟨.program ⟨257⟩, ⟨54675⟩⟩
def mergeEvent : Nat := 162063
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events633.exact162057RawTerms
def group : MergeGroup := .relation 162059
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162059) (rhsResult := 162057)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54672⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 162058 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54672⟩⟩]⟩) (none) 162057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162063

namespace LeftMerge162068
def owner : Owner := ⟨.program ⟨257⟩, ⟨55835⟩⟩
def mergeEvent : Nat := 162068
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩] } }
def leftRaw : List Term := Proof.Events633.exact162064RawTerms
def rightRaw : List Term := Proof.Events632.exact161886RawTerms
def group : MergeGroup := .operator 162064 161886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162064) (leftOrdinal := 0)
    (rightResult := 161886) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55832⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162068

namespace LeftMerge162069
def owner : Owner := ⟨.program ⟨257⟩, ⟨55835⟩⟩
def mergeEvent : Nat := 162069
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨53844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55113⟩⟩] } }
def leftRaw : List Term := Proof.Events633.exact162064RawTerms
def rightRaw : List Term := Proof.Events632.exact161886RawTerms
def group : MergeGroup := .operator 162064 161886
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162064) (leftOrdinal := 2)
    (rightResult := 161886) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨53844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55113⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨53844⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨53844⟩⟩], [⟨.program ⟨257⟩, ⟨55113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162069

namespace LeftMerge162077
def owner : Owner := ⟨.program ⟨257⟩, ⟨55836⟩⟩
def mergeEvent : Nat := 162077
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }
def leftRaw : List Term := Proof.Events633.exact162071RawTerms
def rightRaw : List Term := Proof.Events061.exact15782RawTerms
def group : MergeGroup := .operator 162071 15782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162071) (leftOrdinal := 0)
    (rightResult := 15782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7207⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7207⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162077

namespace LeftMerge162078
def owner : Owner := ⟨.program ⟨257⟩, ⟨55836⟩⟩
def mergeEvent : Nat := 162078
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩] } }
def leftRaw : List Term := Proof.Events633.exact162071RawTerms
def rightRaw : List Term := Proof.Events061.exact15782RawTerms
def group : MergeGroup := .operator 162071 15782
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 162071) (leftOrdinal := 1)
    (rightResult := 15782) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7125⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162078

namespace LeftMerge162080
def owner : Owner := ⟨.program ⟨257⟩, ⟨55836⟩⟩
def mergeEvent : Nat := 162080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15775RawTerms
def group : MergeGroup := .relation 162079
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162079) (rhsResult := 15775)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7125⟩⟩) ⟨7028⟩ 15775) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6757⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162080

namespace LeftMerge162094
def owner : Owner := ⟨.program ⟨257⟩, ⟨52854⟩⟩
def mergeEvent : Nat := 162094
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52852⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155572RawTerms
def rightRaw : List Term := Proof.Events633.exact162088RawTerms
def group : MergeGroup := .operator 155572 162088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155572) (leftOrdinal := 0)
    (rightResult := 162088) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52852⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨52852⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge162094

namespace LeftMerge162095
def owner : Owner := ⟨.program ⟨257⟩, ⟨52854⟩⟩
def mergeEvent : Nat := 162095
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨50864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52852⟩⟩] } }
def leftRaw : List Term := Proof.Events607.exact155572RawTerms
def rightRaw : List Term := Proof.Events633.exact162088RawTerms
def group : MergeGroup := .operator 155572 162088
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 155572) (leftOrdinal := 1)
    (rightResult := 162088) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨50864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52852⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨50864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52852⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162095

namespace LeftMerge162097
def owner : Owner := ⟨.program ⟨257⟩, ⟨52854⟩⟩
def mergeEvent : Nat := 162097
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨50864⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52133⟩⟩] } }
def rhsRaw : List Term := Proof.Events633.exact162085RawTerms
def group : MergeGroup := .relation 162096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 162096) (rhsResult := 162085)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨50864⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨52852⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨52852⟩⟩) ⟨52133⟩ 162085) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52133⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨50864⟩⟩], [⟨.program ⟨257⟩, ⟨52133⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge162097

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
