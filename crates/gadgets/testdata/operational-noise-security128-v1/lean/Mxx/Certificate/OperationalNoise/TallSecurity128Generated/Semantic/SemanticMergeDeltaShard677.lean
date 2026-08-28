import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge112805
def owner : Owner := ⟨.program ⟨257⟩, ⟨23292⟩⟩
def mergeEvent : Nat := 112805
def frameStart : Nat := 112739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events440.exact112801RawTerms
def rightRaw : List Term := Proof.Events440.exact112799RawTerms
def group : MergeGroup := .operator 112801 112799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112801) (leftOrdinal := 0)
    (rightResult := 112799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21816⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112805

namespace LeftMerge112817
def owner : Owner := ⟨.program ⟨257⟩, ⟨23904⟩⟩
def mergeEvent : Nat := 112817
def frameStart : Nat := 112739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩] } }
def leftRaw : List Term := Proof.Events440.exact112813RawTerms
def rightRaw : List Term := Proof.Events440.exact112790RawTerms
def group : MergeGroup := .operator 112813 112790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112813) (leftOrdinal := 0)
    (rightResult := 112790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23903⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112817

namespace LeftMerge112818
def owner : Owner := ⟨.program ⟨257⟩, ⟨23904⟩⟩
def mergeEvent : Nat := 112818
def frameStart : Nat := 112739
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩] } }
def leftRaw : List Term := Proof.Events440.exact112813RawTerms
def rightRaw : List Term := Proof.Events440.exact112790RawTerms
def group : MergeGroup := .operator 112813 112790
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112813) (leftOrdinal := 1)
    (rightResult := 112790) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23903⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21816⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112818

namespace LeftMerge112820
def owner : Owner := ⟨.program ⟨257⟩, ⟨23904⟩⟩
def mergeEvent : Nat := 112820
def frameStart : Nat := 112739
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23090⟩⟩] } }
def rhsRaw : List Term := Proof.Events440.exact112787RawTerms
def group : MergeGroup := .relation 112819
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112819) (rhsResult := 112787)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21816⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨23903⟩⟩) ⟨23090⟩ 112787) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23090⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21816⟩⟩], [⟨.program ⟨257⟩, ⟨23090⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112820

namespace LeftMerge112828
def owner : Owner := ⟨.program ⟨257⟩, ⟨22107⟩⟩
def mergeEvent : Nat := 112828
def frameStart : Nat := 112739
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events440.exact112801RawTerms
def rightRaw : List Term := Proof.Events440.exact112824RawTerms
def group : MergeGroup := .operator 112801 112824
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112801) (leftOrdinal := 0)
    (rightResult := 112824) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22105⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112828

namespace LeftMerge112845
def owner : Owner := ⟨.program ⟨257⟩, ⟨22699⟩⟩
def mergeEvent : Nat := 112845
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }
def rhsRaw : List Term := Proof.Events440.exact112842RawTerms
def group : MergeGroup := .relation 112844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112844) (rhsResult := 112842)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 112843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22696⟩⟩]⟩) (none) 112842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112845

namespace LeftMerge112846
def owner : Owner := ⟨.program ⟨257⟩, ⟨22699⟩⟩
def mergeEvent : Nat := 112846
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩] } }
def rhsRaw : List Term := Proof.Events440.exact112842RawTerms
def group : MergeGroup := .relation 112844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112844) (rhsResult := 112842)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 112843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22696⟩⟩]⟩) (none) 112842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112846

namespace LeftMerge112847
def owner : Owner := ⟨.program ⟨257⟩, ⟨22699⟩⟩
def mergeEvent : Nat := 112847
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23090⟩⟩] } }
def rhsRaw : List Term := Proof.Events440.exact112842RawTerms
def group : MergeGroup := .relation 112844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112844) (rhsResult := 112842)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 112843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22696⟩⟩]⟩) (none) 112842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23090⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨21816⟩⟩], [⟨.program ⟨257⟩, ⟨23090⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112847

namespace LeftMerge112848
def owner : Owner := ⟨.program ⟨257⟩, ⟨22699⟩⟩
def mergeEvent : Nat := 112848
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events440.exact112842RawTerms
def group : MergeGroup := .relation 112844
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 112844) (rhsResult := 112842)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 112843 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22696⟩⟩]⟩) (none) 112842) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22105⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨22105⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112848

namespace LeftMerge112853
def owner : Owner := ⟨.program ⟨257⟩, ⟨23906⟩⟩
def mergeEvent : Nat := 112853
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩] } }
def leftRaw : List Term := Proof.Events440.exact112849RawTerms
def rightRaw : List Term := Proof.Events440.exact112671RawTerms
def group : MergeGroup := .operator 112849 112671
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112849) (leftOrdinal := 0)
    (rightResult := 112671) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨23903⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112853

namespace LeftMerge112854
def owner : Owner := ⟨.program ⟨257⟩, ⟨23906⟩⟩
def mergeEvent : Nat := 112854
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23090⟩⟩] } }
def leftRaw : List Term := Proof.Events440.exact112849RawTerms
def rightRaw : List Term := Proof.Events440.exact112671RawTerms
def group : MergeGroup := .operator 112849 112671
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112849) (leftOrdinal := 2)
    (rightResult := 112671) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23090⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨21816⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23090⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨21816⟩⟩], [⟨.program ⟨257⟩, ⟨23090⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112854

namespace LeftMerge112880
def owner : Owner := ⟨.program ⟨257⟩, ⟨18301⟩⟩
def mergeEvent : Nat := 112880
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events019.exact4950RawTerms
def rightRaw : List Term := Proof.Events410.exact105153RawTerms
def group : MergeGroup := .operator 4950 105153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4950) (leftOrdinal := 0)
    (rightResult := 105153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18298⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112880

namespace LeftMerge112885
def owner : Owner := ⟨.program ⟨257⟩, ⟨8725⟩⟩
def mergeEvent : Nat := 112885
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def leftRaw : List Term := Proof.Events410.exact105023RawTerms
def rightRaw : List Term := Proof.Events098.exact25096RawTerms
def group : MergeGroup := .operator 105023 25096
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105023) (leftOrdinal := 0)
    (rightResult := 25096) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112885

namespace LeftMerge112902
def owner : Owner := ⟨.program ⟨257⟩, ⟨18304⟩⟩
def mergeEvent : Nat := 112902
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12696⟩⟩, ⟨.program ⟨257⟩, ⟨18298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events441.exact112896RawTerms
def rightRaw : List Term := Proof.Events019.exact4953RawTerms
def group : MergeGroup := .operator 112896 4953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112896) (leftOrdinal := 1)
    (rightResult := 4953) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨18298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12696⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12696⟩⟩, ⟨.program ⟨257⟩, ⟨18298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge112902

namespace LeftMerge112903
def owner : Owner := ⟨.program ⟨257⟩, ⟨18304⟩⟩
def mergeEvent : Nat := 112903
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12696⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }
def leftRaw : List Term := Proof.Events441.exact112896RawTerms
def rightRaw : List Term := Proof.Events019.exact4953RawTerms
def group : MergeGroup := .operator 112896 4953
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 112896) (leftOrdinal := 0)
    (rightResult := 4953) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7305⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12696⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12696⟩⟩], [⟨.program ⟨257⟩, ⟨7305⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112903

namespace LeftMerge112908
def owner : Owner := ⟨.program ⟨257⟩, ⟨12697⟩⟩
def mergeEvent : Nat := 112908
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12696⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events019.exact4953RawTerms
def rightRaw : List Term := Proof.Events410.exact105153RawTerms
def group : MergeGroup := .operator 4953 105153
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 4953) (leftOrdinal := 0)
    (rightResult := 105153) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12696⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12696⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge112908

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
