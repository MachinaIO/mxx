import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge129901
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129901
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 6)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129901

namespace LeftMerge129902
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129902
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 5)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129902

namespace LeftMerge129903
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129903
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 4)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129903

namespace LeftMerge129904
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129904
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 3)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129904

namespace LeftMerge129905
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129905
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 2)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129905

namespace LeftMerge129906
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129906
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 1)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129906

namespace LeftMerge129907
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129907
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 0)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge129907

namespace LeftMerge129908
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129908
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 29)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129908

namespace LeftMerge129910
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129910
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129909
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129909) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48311⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129910

namespace LeftMerge129911
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129911
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45631⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 28)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45631⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45631⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129911

namespace LeftMerge129913
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129913
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45631⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129912
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129912) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45631⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45631⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129913

namespace LeftMerge129914
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129914
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42947⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 27)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42947⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42947⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129914

namespace LeftMerge129916
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129916
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42947⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129915
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129915) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42947⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42947⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129916

namespace LeftMerge129917
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129917
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40267⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 26)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40267⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129917

namespace LeftMerge129919
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129919
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40267⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }
def rhsRaw : List Term := Proof.Events506.exact129724RawTerms
def group : MergeGroup := .relation 129918
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 129918) (rhsResult := 129724)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71113⟩⟩) ⟨68806⟩ 129724) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68806⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40267⟩⟩], [⟨.program ⟨257⟩, ⟨68806⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129919

namespace LeftMerge129920
def owner : Owner := ⟨.program ⟨257⟩, ⟨71114⟩⟩
def mergeEvent : Nat := 129920
def frameStart : Nat := 129211
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37591⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩] } }
def leftRaw : List Term := Proof.Events507.exact129886RawTerms
def rightRaw : List Term := Proof.Events506.exact129727RawTerms
def group : MergeGroup := .operator 129886 129727
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 129886) (leftOrdinal := 25)
    (rightResult := 129727) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37591⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37591⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge129920

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
