import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge237528
def owner : Owner := ⟨.program ⟨257⟩, ⟨45892⟩⟩
def mergeEvent : Nat := 237528
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46957⟩⟩] } }
def rhsRaw : List Term := Proof.Events927.exact237524RawTerms
def group : MergeGroup := .relation 237526
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237526) (rhsResult := 237524)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 237525 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45889⟩⟩]⟩) (none) 237524) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46957⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46957⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237528

namespace LeftMerge237529
def owner : Owner := ⟨.program ⟨257⟩, ⟨45892⟩⟩
def mergeEvent : Nat := 237529
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46457⟩⟩] } }
def rhsRaw : List Term := Proof.Events927.exact237524RawTerms
def group : MergeGroup := .relation 237526
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237526) (rhsResult := 237524)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 237525 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45889⟩⟩]⟩) (none) 237524) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46457⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], [⟨.program ⟨257⟩, ⟨46457⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237529

namespace LeftMerge237530
def owner : Owner := ⟨.program ⟨257⟩, ⟨45892⟩⟩
def mergeEvent : Nat := 237530
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events927.exact237524RawTerms
def group : MergeGroup := .relation 237526
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237526) (rhsResult := 237524)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45889⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 237525 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45889⟩⟩]⟩) (none) 237524) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237530

namespace LeftMerge237535
def owner : Owner := ⟨.program ⟨257⟩, ⟨46959⟩⟩
def mergeEvent : Nat := 237535
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46457⟩⟩] } }
def leftRaw : List Term := Proof.Events927.exact237531RawTerms
def rightRaw : List Term := Proof.Events927.exact237345RawTerms
def group : MergeGroup := .operator 237531 237345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237531) (leftOrdinal := 2)
    (rightResult := 237345) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46457⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46457⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨14751⟩⟩, ⟨.program ⟨257⟩, ⟨45106⟩⟩], [⟨.program ⟨257⟩, ⟨46457⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237535

namespace LeftMerge237536
def owner : Owner := ⟨.program ⟨257⟩, ⟨46959⟩⟩
def mergeEvent : Nat := 237536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46957⟩⟩] } }
def leftRaw : List Term := Proof.Events927.exact237531RawTerms
def rightRaw : List Term := Proof.Events927.exact237345RawTerms
def group : MergeGroup := .operator 237531 237345
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237531) (leftOrdinal := 1)
    (rightResult := 237345) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46957⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46957⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨46957⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237536

namespace LeftMerge237544
def owner : Owner := ⟨.program ⟨257⟩, ⟨47301⟩⟩
def mergeEvent : Nat := 237544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩] } }
def leftRaw : List Term := Proof.Events927.exact237538RawTerms
def rightRaw : List Term := Proof.Events926.exact237261RawTerms
def group : MergeGroup := .operator 237538 237261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237538) (leftOrdinal := 0)
    (rightResult := 237261) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47299⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237544

namespace LeftMerge237545
def owner : Owner := ⟨.program ⟨257⟩, ⟨47301⟩⟩
def mergeEvent : Nat := 237545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩] } }
def leftRaw : List Term := Proof.Events927.exact237538RawTerms
def rightRaw : List Term := Proof.Events926.exact237261RawTerms
def group : MergeGroup := .operator 237538 237261
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237538) (leftOrdinal := 1)
    (rightResult := 237261) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47299⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237545

namespace LeftMerge237547
def owner : Owner := ⟨.program ⟨257⟩, ⟨47301⟩⟩
def mergeEvent : Nat := 237547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46603⟩⟩] } }
def rhsRaw : List Term := Proof.Events926.exact237258RawTerms
def group : MergeGroup := .relation 237546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237546) (rhsResult := 237258)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47299⟩⟩) ⟨46603⟩ 237258) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46603⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩, ⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237547

namespace LeftMerge237561
def owner : Owner := ⟨.program ⟨257⟩, ⟨46179⟩⟩
def mergeEvent : Nat := 237561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩] } }
def leftRaw : List Term := Proof.Events925.exact236870RawTerms
def rightRaw : List Term := Proof.Events927.exact237555RawTerms
def group : MergeGroup := .operator 236870 237555
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 236870) (leftOrdinal := 0)
    (rightResult := 237555) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46176⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237561

namespace LeftMerge237682
def owner : Owner := ⟨.program ⟨257⟩, ⟨46820⟩⟩
def mergeEvent : Nat := 237682
def frameStart : Nat := 237616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events928.exact237678RawTerms
def rightRaw : List Term := Proof.Events928.exact237676RawTerms
def group : MergeGroup := .operator 237678 237676
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237678) (leftOrdinal := 0)
    (rightResult := 237676) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237682

namespace LeftMerge237694
def owner : Owner := ⟨.program ⟨257⟩, ⟨47300⟩⟩
def mergeEvent : Nat := 237694
def frameStart : Nat := 237616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩] } }
def leftRaw : List Term := Proof.Events928.exact237690RawTerms
def rightRaw : List Term := Proof.Events928.exact237667RawTerms
def group : MergeGroup := .operator 237690 237667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237690) (leftOrdinal := 0)
    (rightResult := 237667) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47299⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237694

namespace LeftMerge237695
def owner : Owner := ⟨.program ⟨257⟩, ⟨47300⟩⟩
def mergeEvent : Nat := 237695
def frameStart : Nat := 237616
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩] } }
def leftRaw : List Term := Proof.Events928.exact237690RawTerms
def rightRaw : List Term := Proof.Events928.exact237667RawTerms
def group : MergeGroup := .operator 237690 237667
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237690) (leftOrdinal := 1)
    (rightResult := 237667) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47299⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237695

namespace LeftMerge237697
def owner : Owner := ⟨.program ⟨257⟩, ⟨47300⟩⟩
def mergeEvent : Nat := 237697
def frameStart : Nat := 237616
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45452⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46603⟩⟩] } }
def rhsRaw : List Term := Proof.Events928.exact237664RawTerms
def group : MergeGroup := .relation 237696
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237696) (rhsResult := 237664)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47299⟩⟩) ⟨46603⟩ 237664) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46603⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45452⟩⟩], [⟨.program ⟨257⟩, ⟨46603⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237697

namespace LeftMerge237705
def owner : Owner := ⟨.program ⟨257⟩, ⟨45658⟩⟩
def mergeEvent : Nat := 237705
def frameStart : Nat := 237616
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45657⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events928.exact237678RawTerms
def rightRaw : List Term := Proof.Events928.exact237701RawTerms
def group : MergeGroup := .operator 237678 237701
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 237678) (leftOrdinal := 0)
    (rightResult := 237701) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45657⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45657⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237705

namespace LeftMerge237722
def owner : Owner := ⟨.program ⟨257⟩, ⟨46179⟩⟩
def mergeEvent : Nat := 237722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }
def rhsRaw : List Term := Proof.Events928.exact237719RawTerms
def group : MergeGroup := .relation 237721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237721) (rhsResult := 237719)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 237720 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩) (none) 237719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7230⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7230⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge237722

namespace LeftMerge237723
def owner : Owner := ⟨.program ⟨257⟩, ⟨46179⟩⟩
def mergeEvent : Nat := 237723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4993⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩] } }
def rhsRaw : List Term := Proof.Events928.exact237719RawTerms
def group : MergeGroup := .relation 237721
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 237721) (rhsResult := 237719)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 237720 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨46176⟩⟩]⟩) (none) 237719) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4993⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47299⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge237723

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
