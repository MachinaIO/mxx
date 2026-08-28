import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge271989
def owner : Owner := ⟨.program ⟨257⟩, ⟨53301⟩⟩
def mergeEvent : Nat := 271989
def frameStart : Nat := 271959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1062.exact271985RawTerms
def rightRaw : List Term := Proof.Events1062.exact271982RawTerms
def group : MergeGroup := .operator 271985 271982
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 271985) (leftOrdinal := 0)
    (rightResult := 271982) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24670⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge271989

namespace LeftMerge272019
def owner : Owner := ⟨.program ⟨257⟩, ⟨55236⟩⟩
def mergeEvent : Nat := 272019
def frameStart : Nat := 271959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1062.exact272015RawTerms
def rightRaw : List Term := Proof.Events1062.exact272013RawTerms
def group : MergeGroup := .operator 272015 272013
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272015) (leftOrdinal := 0)
    (rightResult := 272013) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272019

namespace LeftMerge272042
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def mergeEvent : Nat := 272042
def frameStart : Nat := 271959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }
def leftRaw : List Term := Proof.Events1062.exact272038RawTerms
def rightRaw : List Term := Proof.Events1062.exact272035RawTerms
def group : MergeGroup := .operator 272038 272035
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272038) (leftOrdinal := 0)
    (rightResult := 272035) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9529⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272042

namespace LeftMerge272051
def owner : Owner := ⟨.program ⟨257⟩, ⟨55411⟩⟩
def mergeEvent : Nat := 272051
def frameStart : Nat := 271959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩] } }
def leftRaw : List Term := Proof.Events1062.exact272047RawTerms
def rightRaw : List Term := Proof.Events1062.exact272004RawTerms
def group : MergeGroup := .operator 272047 272004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272047) (leftOrdinal := 0)
    (rightResult := 272004) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55408⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272051

namespace LeftMerge272052
def owner : Owner := ⟨.program ⟨257⟩, ⟨55411⟩⟩
def mergeEvent : Nat := 272052
def frameStart : Nat := 271959
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩] } }
def leftRaw : List Term := Proof.Events1062.exact272047RawTerms
def rightRaw : List Term := Proof.Events1062.exact272004RawTerms
def group : MergeGroup := .operator 272047 272004
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272047) (leftOrdinal := 1)
    (rightResult := 272004) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55408⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272052

namespace LeftMerge272054
def owner : Owner := ⟨.program ⟨257⟩, ⟨55411⟩⟩
def mergeEvent : Nat := 272054
def frameStart : Nat := 271959
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54939⟩⟩] } }
def rhsRaw : List Term := Proof.Events1062.exact272001RawTerms
def group : MergeGroup := .relation 272053
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272053) (rhsResult := 272001)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55408⟩⟩) ⟨54939⟩ 272001) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨54939⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], [⟨.program ⟨257⟩, ⟨54939⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272054

namespace LeftMerge272062
def owner : Owner := ⟨.program ⟨257⟩, ⟨53804⟩⟩
def mergeEvent : Nat := 272062
def frameStart : Nat := 271959
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1062.exact272015RawTerms
def rightRaw : List Term := Proof.Events1062.exact272058RawTerms
def group : MergeGroup := .operator 272015 272058
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272015) (leftOrdinal := 0)
    (rightResult := 272058) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272062

namespace LeftMerge272079
def owner : Owner := ⟨.program ⟨257⟩, ⟨54349⟩⟩
def mergeEvent : Nat := 272079
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }
def rhsRaw : List Term := Proof.Events1062.exact272076RawTerms
def group : MergeGroup := .relation 272078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272078) (rhsResult := 272076)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54346⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54346⟩⟩]⟩) (none) 272076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272079

namespace LeftMerge272080
def owner : Owner := ⟨.program ⟨257⟩, ⟨54349⟩⟩
def mergeEvent : Nat := 272080
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩] } }
def rhsRaw : List Term := Proof.Events1062.exact272076RawTerms
def group : MergeGroup := .relation 272078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272078) (rhsResult := 272076)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54346⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54346⟩⟩]⟩) (none) 272076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272080

namespace LeftMerge272081
def owner : Owner := ⟨.program ⟨257⟩, ⟨54349⟩⟩
def mergeEvent : Nat := 272081
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54939⟩⟩] } }
def rhsRaw : List Term := Proof.Events1062.exact272076RawTerms
def group : MergeGroup := .relation 272078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272078) (rhsResult := 272076)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54346⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54346⟩⟩]⟩) (none) 272076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54939⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], [⟨.program ⟨257⟩, ⟨54939⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272081

namespace LeftMerge272082
def owner : Owner := ⟨.program ⟨257⟩, ⟨54349⟩⟩
def mergeEvent : Nat := 272082
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1062.exact272076RawTerms
def group : MergeGroup := .relation 272078
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272078) (rhsResult := 272076)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54346⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 272077 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨54346⟩⟩]⟩) (none) 272076) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272082

namespace LeftMerge272087
def owner : Owner := ⟨.program ⟨257⟩, ⟨55410⟩⟩
def mergeEvent : Nat := 272087
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54939⟩⟩] } }
def leftRaw : List Term := Proof.Events1062.exact272083RawTerms
def rightRaw : List Term := Proof.Events1062.exact271897RawTerms
def group : MergeGroup := .operator 272083 271897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272083) (leftOrdinal := 2)
    (rightResult := 271897) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54939⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨54939⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨24670⟩⟩, ⟨.program ⟨257⟩, ⟨53300⟩⟩], [⟨.program ⟨257⟩, ⟨54939⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272087

namespace LeftMerge272088
def owner : Owner := ⟨.program ⟨257⟩, ⟨55410⟩⟩
def mergeEvent : Nat := 272088
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩] } }
def leftRaw : List Term := Proof.Events1062.exact272083RawTerms
def rightRaw : List Term := Proof.Events1062.exact271897RawTerms
def group : MergeGroup := .operator 272083 271897
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272083) (leftOrdinal := 1)
    (rightResult := 271897) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7289⟩⟩, ⟨.program ⟨257⟩, ⟨9529⟩⟩, ⟨.program ⟨257⟩, ⟨55408⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272088

namespace LeftMerge272096
def owner : Owner := ⟨.program ⟨257⟩, ⟨55677⟩⟩
def mergeEvent : Nat := 272096
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩] } }
def leftRaw : List Term := Proof.Events1062.exact272090RawTerms
def rightRaw : List Term := Proof.Events1061.exact271813RawTerms
def group : MergeGroup := .operator 272090 271813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272090) (leftOrdinal := 0)
    (rightResult := 271813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7184⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7184⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge272096

namespace LeftMerge272097
def owner : Owner := ⟨.program ⟨257⟩, ⟨55677⟩⟩
def mergeEvent : Nat := 272097
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩] } }
def leftRaw : List Term := Proof.Events1062.exact272090RawTerms
def rightRaw : List Term := Proof.Events1061.exact271813RawTerms
def group : MergeGroup := .operator 272090 271813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 272090) (leftOrdinal := 1)
    (rightResult := 271813) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55675⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272097

namespace LeftMerge272099
def owner : Owner := ⟨.program ⟨257⟩, ⟨55677⟩⟩
def mergeEvent : Nat := 272099
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨55066⟩⟩] } }
def rhsRaw : List Term := Proof.Events1061.exact271810RawTerms
def group : MergeGroup := .relation 272098
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 272098) (rhsResult := 271810)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨55675⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨55675⟩⟩) ⟨55066⟩ 271810) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨55066⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨53802⟩⟩], [⟨.program ⟨257⟩, ⟨55066⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge272099

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
