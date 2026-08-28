import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge54640
def owner : Owner := ⟨.program ⟨257⟩, ⟨20309⟩⟩
def mergeEvent : Nat := 54640
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12801⟩⟩, ⟨.program ⟨257⟩, ⟨18466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19757⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54636RawTerms
def rightRaw : List Term := Proof.Events212.exact54450RawTerms
def group : MergeGroup := .operator 54636 54450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54636) (leftOrdinal := 2)
    (rightResult := 54450) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12801⟩⟩, ⟨.program ⟨257⟩, ⟨18466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19757⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12801⟩⟩, ⟨.program ⟨257⟩, ⟨18466⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19757⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨12801⟩⟩, ⟨.program ⟨257⟩, ⟨18466⟩⟩], [⟨.program ⟨257⟩, ⟨19757⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54640

namespace LeftMerge54641
def owner : Owner := ⟨.program ⟨257⟩, ⟨20309⟩⟩
def mergeEvent : Nat := 54641
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20307⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54636RawTerms
def rightRaw : List Term := Proof.Events212.exact54450RawTerms
def group : MergeGroup := .operator 54636 54450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54636) (leftOrdinal := 1)
    (rightResult := 54450) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20307⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20307⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7277⟩⟩, ⟨.program ⟨257⟩, ⟨9571⟩⟩, ⟨.program ⟨257⟩, ⟨20307⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54641

namespace LeftMerge54649
def owner : Owner := ⟨.program ⟨257⟩, ⟨20902⟩⟩
def mergeEvent : Nat := 54649
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54643RawTerms
def rightRaw : List Term := Proof.Events212.exact54366RawTerms
def group : MergeGroup := .operator 54643 54366
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54643) (leftOrdinal := 0)
    (rightResult := 54366) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20900⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54649

namespace LeftMerge54650
def owner : Owner := ⟨.program ⟨257⟩, ⟨20902⟩⟩
def mergeEvent : Nat := 54650
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54643RawTerms
def rightRaw : List Term := Proof.Events212.exact54366RawTerms
def group : MergeGroup := .operator 54643 54366
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54643) (leftOrdinal := 1)
    (rightResult := 54366) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20900⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54650

namespace LeftMerge54652
def owner : Owner := ⟨.program ⟨257⟩, ⟨20902⟩⟩
def mergeEvent : Nat := 54652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19933⟩⟩] } }
def rhsRaw : List Term := Proof.Events212.exact54363RawTerms
def group : MergeGroup := .relation 54651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54651) (rhsResult := 54363)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20900⟩⟩) ⟨19933⟩ 54363) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19933⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨19933⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54652

namespace LeftMerge54666
def owner : Owner := ⟨.program ⟨257⟩, ⟨19619⟩⟩
def mergeEvent : Nat := 54666
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46745RawTerms
def rightRaw : List Term := Proof.Events213.exact54660RawTerms
def group : MergeGroup := .operator 46745 54660
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46745) (leftOrdinal := 0)
    (rightResult := 54660) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19616⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54666

namespace LeftMerge54787
def owner : Owner := ⟨.program ⟨257⟩, ⟨20100⟩⟩
def mergeEvent : Nat := 54787
def frameStart : Nat := 54721
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54783RawTerms
def rightRaw : List Term := Proof.Events213.exact54781RawTerms
def group : MergeGroup := .operator 54783 54781
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54783) (leftOrdinal := 0)
    (rightResult := 54781) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54787

namespace LeftMerge54799
def owner : Owner := ⟨.program ⟨257⟩, ⟨20901⟩⟩
def mergeEvent : Nat := 54799
def frameStart : Nat := 54721
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩] } }
def leftRaw : List Term := Proof.Events214.exact54795RawTerms
def rightRaw : List Term := Proof.Events213.exact54772RawTerms
def group : MergeGroup := .operator 54795 54772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54795) (leftOrdinal := 0)
    (rightResult := 54772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20900⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54799

namespace LeftMerge54800
def owner : Owner := ⟨.program ⟨257⟩, ⟨20901⟩⟩
def mergeEvent : Nat := 54800
def frameStart : Nat := 54721
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩] } }
def leftRaw : List Term := Proof.Events214.exact54795RawTerms
def rightRaw : List Term := Proof.Events213.exact54772RawTerms
def group : MergeGroup := .operator 54795 54772
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54795) (leftOrdinal := 1)
    (rightResult := 54772) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨20900⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54800

namespace LeftMerge54802
def owner : Owner := ⟨.program ⟨257⟩, ⟨20901⟩⟩
def mergeEvent : Nat := 54802
def frameStart : Nat := 54721
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19933⟩⟩] } }
def rhsRaw : List Term := Proof.Events213.exact54769RawTerms
def group : MergeGroup := .relation 54801
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54801) (rhsResult := 54769)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨20900⟩⟩) ⟨19933⟩ 54769) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨19933⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨19933⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54802

namespace LeftMerge54810
def owner : Owner := ⟨.program ⟨257⟩, ⟨19020⟩⟩
def mergeEvent : Nat := 54810
def frameStart : Nat := 54721
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events213.exact54783RawTerms
def rightRaw : List Term := Proof.Events214.exact54806RawTerms
def group : MergeGroup := .operator 54783 54806
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54783) (leftOrdinal := 0)
    (rightResult := 54806) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54810

namespace LeftMerge54827
def owner : Owner := ⟨.program ⟨257⟩, ⟨19619⟩⟩
def mergeEvent : Nat := 54827
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }
def rhsRaw : List Term := Proof.Events214.exact54824RawTerms
def group : MergeGroup := .relation 54826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54826) (rhsResult := 54824)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54825 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩]⟩) (none) 54824) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7200⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54827

namespace LeftMerge54828
def owner : Owner := ⟨.program ⟨257⟩, ⟨19619⟩⟩
def mergeEvent : Nat := 54828
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩] } }
def rhsRaw : List Term := Proof.Events214.exact54824RawTerms
def group : MergeGroup := .relation 54826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54826) (rhsResult := 54824)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54825 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩]⟩) (none) 54824) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54828

namespace LeftMerge54829
def owner : Owner := ⟨.program ⟨257⟩, ⟨19619⟩⟩
def mergeEvent : Nat := 54829
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19933⟩⟩] } }
def rhsRaw : List Term := Proof.Events214.exact54824RawTerms
def group : MergeGroup := .relation 54826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54826) (rhsResult := 54824)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54825 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩]⟩) (none) 54824) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18652⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨19933⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨18652⟩⟩], [⟨.program ⟨257⟩, ⟨19933⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54829

namespace LeftMerge54830
def owner : Owner := ⟨.program ⟨257⟩, ⟨19619⟩⟩
def mergeEvent : Nat := 54830
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events214.exact54824RawTerms
def group : MergeGroup := .relation 54826
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 54826) (rhsResult := 54824)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 54825 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨19616⟩⟩]⟩) (none) 54824) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨19018⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨19018⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge54830

namespace LeftMerge54835
def owner : Owner := ⟨.program ⟨257⟩, ⟨20903⟩⟩
def mergeEvent : Nat := 54835
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩] } }
def leftRaw : List Term := Proof.Events214.exact54831RawTerms
def rightRaw : List Term := Proof.Events213.exact54653RawTerms
def group : MergeGroup := .operator 54831 54653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 54831) (leftOrdinal := 0)
    (rightResult := 54653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7180⟩⟩, ⟨.program ⟨257⟩, ⟨20900⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge54835

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
