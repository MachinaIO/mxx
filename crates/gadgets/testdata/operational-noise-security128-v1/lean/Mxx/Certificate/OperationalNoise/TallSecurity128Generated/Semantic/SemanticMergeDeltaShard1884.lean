import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge304456
def owner : Owner := ⟨.program ⟨257⟩, ⟨68273⟩⟩
def mergeEvent : Nat := 304456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1189.exact304420RawTerms
def group : MergeGroup := .relation 304422
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304422) (rhsResult := 304420)
    (sourceTermOrdinal := 24) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 304421 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩) (none) 304420) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31916⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨31916⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304456

namespace LeftMerge304457
def owner : Owner := ⟨.program ⟨257⟩, ⟨68273⟩⟩
def mergeEvent : Nat := 304457
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1189.exact304420RawTerms
def group : MergeGroup := .relation 304422
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304422) (rhsResult := 304420)
    (sourceTermOrdinal := 21) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 304421 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩) (none) 304420) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨21896⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304457

namespace LeftMerge304458
def owner : Owner := ⟨.program ⟨257⟩, ⟨68273⟩⟩
def mergeEvent : Nat := 304458
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1189.exact304420RawTerms
def group : MergeGroup := .relation 304422
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304422) (rhsResult := 304420)
    (sourceTermOrdinal := 20) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 304421 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩) (none) 304420) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18676⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨18676⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304458

namespace LeftMerge304459
def owner : Owner := ⟨.program ⟨257⟩, ⟨68273⟩⟩
def mergeEvent : Nat := 304459
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1189.exact304420RawTerms
def group : MergeGroup := .relation 304422
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304422) (rhsResult := 304420)
    (sourceTermOrdinal := 19) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 304421 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩) (none) 304420) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15875⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨15875⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304459

namespace LeftMerge304460
def owner : Owner := ⟨.program ⟨257⟩, ⟨68273⟩⟩
def mergeEvent : Nat := 304460
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨67271⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1189.exact304420RawTerms
def group : MergeGroup := .relation 304422
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 304422) (rhsResult := 304420)
    (sourceTermOrdinal := 37) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 304421 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨68270⟩⟩]⟩) (none) 304420) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨67271⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨67271⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304460

namespace LeftMerge304465
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304465
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 17)
    (rightResult := 303069) (rightOrdinal := 17) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304465

namespace LeftMerge304466
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304466
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 30)
    (rightResult := 303069) (rightOrdinal := 29) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304466

namespace LeftMerge304467
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304467
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 16)
    (rightResult := 303069) (rightOrdinal := 16) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304467

namespace LeftMerge304468
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 29)
    (rightResult := 303069) (rightOrdinal := 28) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304468

namespace LeftMerge304469
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304469
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 15)
    (rightResult := 303069) (rightOrdinal := 15) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304469

namespace LeftMerge304470
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304470
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 28)
    (rightResult := 303069) (rightOrdinal := 27) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304470

namespace LeftMerge304471
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 14)
    (rightResult := 303069) (rightOrdinal := 14) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304471

namespace LeftMerge304472
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304472
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 27)
    (rightResult := 303069) (rightOrdinal := 26) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304472

namespace LeftMerge304473
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304473
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 13)
    (rightResult := 303069) (rightOrdinal := 13) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304473

namespace LeftMerge304474
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304474
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 26)
    (rightResult := 303069) (rightOrdinal := 25) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge304474

namespace LeftMerge304475
def owner : Owner := ⟨.program ⟨257⟩, ⟨70937⟩⟩
def mergeEvent : Nat := 304475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1189.exact304461RawTerms
def rightRaw : List Term := Proof.Events1183.exact303069RawTerms
def group : MergeGroup := .operator 304461 303069
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 304461) (leftOrdinal := 12)
    (rightResult := 303069) (rightOrdinal := 12) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge304475

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
