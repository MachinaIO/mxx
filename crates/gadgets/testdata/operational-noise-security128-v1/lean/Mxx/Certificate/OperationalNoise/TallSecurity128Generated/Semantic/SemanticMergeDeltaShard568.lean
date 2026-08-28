import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge94934
def owner : Owner := ⟨.program ⟨257⟩, ⟨62607⟩⟩
def mergeEvent : Nat := 94934
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events370.exact94925RawTerms
def rightRaw : List Term := Proof.Events084.exact21619RawTerms
def group : MergeGroup := .operator 94925 21619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94925) (leftOrdinal := 0)
    (rightResult := 21619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94934

namespace LeftMerge94939
def owner : Owner := ⟨.program ⟨257⟩, ⟨62608⟩⟩
def mergeEvent : Nat := 94939
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }
def leftRaw : List Term := Proof.Events370.exact94935RawTerms
def rightRaw : List Term := Proof.Events370.exact94905RawTerms
def group : MergeGroup := .operator 94935 94905
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94935) (leftOrdinal := 1)
    (rightResult := 94905) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7275⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], [⟨.program ⟨257⟩, ⟨7275⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94939

namespace LeftMerge94947
def owner : Owner := ⟨.program ⟨257⟩, ⟨64495⟩⟩
def mergeEvent : Nat := 94947
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩] } }
def leftRaw : List Term := Proof.Events370.exact94941RawTerms
def rightRaw : List Term := Proof.Events370.exact94877RawTerms
def group : MergeGroup := .operator 94941 94877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94941) (leftOrdinal := 1)
    (rightResult := 94877) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64494⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94947

namespace LeftMerge94949
def owner : Owner := ⟨.program ⟨257⟩, ⟨64495⟩⟩
def mergeEvent : Nat := 94949
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63959⟩⟩] } }
def rhsRaw : List Term := Proof.Events370.exact94874RawTerms
def group : MergeGroup := .relation 94948
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 94948) (rhsResult := 94874)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64494⟩⟩) ⟨63959⟩ 94874) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], [⟨.program ⟨257⟩, ⟨63959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge94949

namespace LeftMerge94950
def owner : Owner := ⟨.program ⟨257⟩, ⟨64495⟩⟩
def mergeEvent : Nat := 94950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩] } }
def leftRaw : List Term := Proof.Events370.exact94941RawTerms
def rightRaw : List Term := Proof.Events370.exact94877RawTerms
def group : MergeGroup := .operator 94941 94877
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 94941) (leftOrdinal := 0)
    (rightResult := 94877) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64494⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94950

namespace LeftMerge94964
def owner : Owner := ⟨.program ⟨257⟩, ⟨63422⟩⟩
def mergeEvent : Nat := 94964
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63419⟩⟩] } }
def leftRaw : List Term := Proof.Events353.exact90620RawTerms
def rightRaw : List Term := Proof.Events370.exact94958RawTerms
def group : MergeGroup := .operator 90620 94958
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 90620) (leftOrdinal := 0)
    (rightResult := 94958) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63419⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63419⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge94964

namespace LeftMerge95043
def owner : Owner := ⟨.program ⟨257⟩, ⟨62601⟩⟩
def mergeEvent : Nat := 95043
def frameStart : Nat := 95013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events371.exact95039RawTerms
def rightRaw : List Term := Proof.Events371.exact95036RawTerms
def group : MergeGroup := .operator 95039 95036
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95039) (leftOrdinal := 0)
    (rightResult := 95036) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25550⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95043

namespace LeftMerge95073
def owner : Owner := ⟨.program ⟨257⟩, ⟨64228⟩⟩
def mergeEvent : Nat := 95073
def frameStart : Nat := 95013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events371.exact95069RawTerms
def rightRaw : List Term := Proof.Events371.exact95067RawTerms
def group : MergeGroup := .operator 95069 95067
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95069) (leftOrdinal := 0)
    (rightResult := 95067) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95073

namespace LeftMerge95096
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 95096
def frameStart : Nat := 95013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events371.exact95092RawTerms
def rightRaw : List Term := Proof.Events371.exact95089RawTerms
def group : MergeGroup := .operator 95092 95089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95092) (leftOrdinal := 0)
    (rightResult := 95089) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95096

namespace LeftMerge95105
def owner : Owner := ⟨.program ⟨257⟩, ⟨64497⟩⟩
def mergeEvent : Nat := 95105
def frameStart : Nat := 95013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩] } }
def leftRaw : List Term := Proof.Events371.exact95101RawTerms
def rightRaw : List Term := Proof.Events371.exact95058RawTerms
def group : MergeGroup := .operator 95101 95058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95101) (leftOrdinal := 0)
    (rightResult := 95058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64494⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95105

namespace LeftMerge95106
def owner : Owner := ⟨.program ⟨257⟩, ⟨64497⟩⟩
def mergeEvent : Nat := 95106
def frameStart : Nat := 95013
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩] } }
def leftRaw : List Term := Proof.Events371.exact95101RawTerms
def rightRaw : List Term := Proof.Events371.exact95058RawTerms
def group : MergeGroup := .operator 95101 95058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95101) (leftOrdinal := 1)
    (rightResult := 95058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64494⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95106

namespace LeftMerge95108
def owner : Owner := ⟨.program ⟨257⟩, ⟨64497⟩⟩
def mergeEvent : Nat := 95108
def frameStart : Nat := 95013
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63959⟩⟩] } }
def rhsRaw : List Term := Proof.Events371.exact95055RawTerms
def group : MergeGroup := .relation 95107
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95107) (rhsResult := 95055)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64494⟩⟩) ⟨63959⟩ 95055) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], [⟨.program ⟨257⟩, ⟨63959⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95108

namespace LeftMerge95116
def owner : Owner := ⟨.program ⟨257⟩, ⟨62850⟩⟩
def mergeEvent : Nat := 95116
def frameStart : Nat := 95013
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62848⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events371.exact95069RawTerms
def rightRaw : List Term := Proof.Events371.exact95112RawTerms
def group : MergeGroup := .operator 95069 95112
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 95069) (leftOrdinal := 0)
    (rightResult := 95112) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62848⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62848⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95116

namespace LeftMerge95133
def owner : Owner := ⟨.program ⟨257⟩, ⟨63422⟩⟩
def mergeEvent : Nat := 95133
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events371.exact95130RawTerms
def group : MergeGroup := .relation 95132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95132) (rhsResult := 95130)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95131 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63419⟩⟩]⟩) (none) 95130) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95133

namespace LeftMerge95134
def owner : Owner := ⟨.program ⟨257⟩, ⟨63422⟩⟩
def mergeEvent : Nat := 95134
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩] } }
def rhsRaw : List Term := Proof.Events371.exact95130RawTerms
def group : MergeGroup := .relation 95132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95132) (rhsResult := 95130)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95131 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63419⟩⟩]⟩) (none) 95130) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64494⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge95134

namespace LeftMerge95135
def owner : Owner := ⟨.program ⟨257⟩, ⟨63422⟩⟩
def mergeEvent : Nat := 95135
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63959⟩⟩] } }
def rhsRaw : List Term := Proof.Events371.exact95130RawTerms
def group : MergeGroup := .relation 95132
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 95132) (rhsResult := 95130)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63419⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 95131 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63419⟩⟩]⟩) (none) 95130) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63959⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10270⟩⟩, ⟨.program ⟨257⟩, ⟨25550⟩⟩, ⟨.program ⟨257⟩, ⟨62600⟩⟩], [⟨.program ⟨257⟩, ⟨63959⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge95135

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
