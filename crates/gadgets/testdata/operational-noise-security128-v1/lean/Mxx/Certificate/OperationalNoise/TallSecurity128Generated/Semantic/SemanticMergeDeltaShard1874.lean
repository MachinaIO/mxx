import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge302998
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 302998
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 29)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge302998

namespace LeftMerge303000
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303000
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1152.exact295080RawTerms
def group : MergeGroup := .relation 302999
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 302999) (rhsResult := 295080)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨48233⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303000

namespace LeftMerge303001
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303001
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 16)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge303001

namespace LeftMerge303002
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303002
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 28)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303002

namespace LeftMerge303004
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303004
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1152.exact295080RawTerms
def group : MergeGroup := .relation 303003
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 303003) (rhsResult := 295080)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨45553⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303004

namespace LeftMerge303005
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303005
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 15)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge303005

namespace LeftMerge303006
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303006
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 27)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303006

namespace LeftMerge303008
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303008
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1152.exact295080RawTerms
def group : MergeGroup := .relation 303007
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 303007) (rhsResult := 295080)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨42869⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303008

namespace LeftMerge303009
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303009
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 14)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge303009

namespace LeftMerge303010
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303010
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 26)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303010

namespace LeftMerge303012
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303012
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1152.exact295080RawTerms
def group : MergeGroup := .relation 303011
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 303011) (rhsResult := 295080)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨40189⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303012

namespace LeftMerge303013
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303013
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 13)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge303013

namespace LeftMerge303014
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303014
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 25)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303014

namespace LeftMerge303016
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303016
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }
def rhsRaw : List Term := Proof.Events1152.exact295080RawTerms
def group : MergeGroup := .relation 303015
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 303015) (rhsResult := 295080)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70934⟩⟩) ⟨68770⟩ 295080) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68770⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨37513⟩⟩], [⟨.program ⟨257⟩, ⟨68770⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303016

namespace LeftMerge303017
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303017
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 12)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge303017

namespace LeftMerge303018
def owner : Owner := ⟨.program ⟨257⟩, ⟨70936⟩⟩
def mergeEvent : Nat := 303018
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩] } }
def leftRaw : List Term := Proof.Events1183.exact302991RawTerms
def rightRaw : List Term := Proof.Events1152.exact295083RawTerms
def group : MergeGroup := .operator 302991 295083
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 302991) (leftOrdinal := 24)
    (rightResult := 295083) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70934⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨34833⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70934⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge303018

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
