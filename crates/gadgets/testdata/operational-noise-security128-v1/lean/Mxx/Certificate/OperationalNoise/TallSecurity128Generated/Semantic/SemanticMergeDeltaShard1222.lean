import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge199436
def owner : Owner := ⟨.program ⟨257⟩, ⟨51472⟩⟩
def mergeEvent : Nat := 199436
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }
def rhsRaw : List Term := Proof.Events779.exact199433RawTerms
def group : MergeGroup := .relation 199435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199435) (rhsResult := 199433)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 199434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51469⟩⟩]⟩) (none) 199433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199436

namespace LeftMerge199437
def owner : Owner := ⟨.program ⟨257⟩, ⟨51472⟩⟩
def mergeEvent : Nat := 199437
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52541⟩⟩] } }
def rhsRaw : List Term := Proof.Events779.exact199433RawTerms
def group : MergeGroup := .relation 199435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199435) (rhsResult := 199433)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 199434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51469⟩⟩]⟩) (none) 199433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52541⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52541⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199437

namespace LeftMerge199438
def owner : Owner := ⟨.program ⟨257⟩, ⟨51472⟩⟩
def mergeEvent : Nat := 199438
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24554⟩⟩, ⟨.program ⟨257⟩, ⟨50599⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52021⟩⟩] } }
def rhsRaw : List Term := Proof.Events779.exact199433RawTerms
def group : MergeGroup := .relation 199435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199435) (rhsResult := 199433)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 199434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51469⟩⟩]⟩) (none) 199433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨24554⟩⟩, ⟨.program ⟨257⟩, ⟨50599⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52021⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24554⟩⟩, ⟨.program ⟨257⟩, ⟨50599⟩⟩], [⟨.program ⟨257⟩, ⟨52021⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199438

namespace LeftMerge199439
def owner : Owner := ⟨.program ⟨257⟩, ⟨51472⟩⟩
def mergeEvent : Nat := 199439
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events779.exact199433RawTerms
def group : MergeGroup := .relation 199435
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199435) (rhsResult := 199433)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51469⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 199434 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51469⟩⟩]⟩) (none) 199433) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199439

namespace LeftMerge199444
def owner : Owner := ⟨.program ⟨257⟩, ⟨52543⟩⟩
def mergeEvent : Nat := 199444
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24554⟩⟩, ⟨.program ⟨257⟩, ⟨50599⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52021⟩⟩] } }
def leftRaw : List Term := Proof.Events779.exact199440RawTerms
def rightRaw : List Term := Proof.Events778.exact199254RawTerms
def group : MergeGroup := .operator 199440 199254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199440) (leftOrdinal := 2)
    (rightResult := 199254) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24554⟩⟩, ⟨.program ⟨257⟩, ⟨50599⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52021⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24554⟩⟩, ⟨.program ⟨257⟩, ⟨50599⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52021⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨24554⟩⟩, ⟨.program ⟨257⟩, ⟨50599⟩⟩], [⟨.program ⟨257⟩, ⟨52021⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199444

namespace LeftMerge199445
def owner : Owner := ⟨.program ⟨257⟩, ⟨52543⟩⟩
def mergeEvent : Nat := 199445
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52541⟩⟩] } }
def leftRaw : List Term := Proof.Events779.exact199440RawTerms
def rightRaw : List Term := Proof.Events778.exact199254RawTerms
def group : MergeGroup := .operator 199440 199254
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199440) (leftOrdinal := 1)
    (rightResult := 199254) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52541⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52541⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7288⟩⟩, ⟨.program ⟨257⟩, ⟨9580⟩⟩, ⟨.program ⟨257⟩, ⟨52541⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199445

namespace LeftMerge199453
def owner : Owner := ⟨.program ⟨257⟩, ⟨53016⟩⟩
def mergeEvent : Nat := 199453
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩] } }
def leftRaw : List Term := Proof.Events779.exact199447RawTerms
def rightRaw : List Term := Proof.Events778.exact199170RawTerms
def group : MergeGroup := .operator 199447 199170
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199447) (leftOrdinal := 0)
    (rightResult := 199170) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53014⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199453

namespace LeftMerge199454
def owner : Owner := ⟨.program ⟨257⟩, ⟨53016⟩⟩
def mergeEvent : Nat := 199454
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩] } }
def leftRaw : List Term := Proof.Events779.exact199447RawTerms
def rightRaw : List Term := Proof.Events778.exact199170RawTerms
def group : MergeGroup := .operator 199447 199170
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199447) (leftOrdinal := 1)
    (rightResult := 199170) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53014⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199454

namespace LeftMerge199456
def owner : Owner := ⟨.program ⟨257⟩, ⟨53016⟩⟩
def mergeEvent : Nat := 199456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52179⟩⟩] } }
def rhsRaw : List Term := Proof.Events777.exact199167RawTerms
def group : MergeGroup := .relation 199455
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199455) (rhsResult := 199167)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53014⟩⟩) ⟨52179⟩ 199167) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨52179⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199456

namespace LeftMerge199470
def owner : Owner := ⟨.program ⟨257⟩, ⟨51799⟩⟩
def mergeEvent : Nat := 199470
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51796⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events779.exact199464RawTerms
def group : MergeGroup := .operator 192995 199464
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 199464) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51796⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51796⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199470

namespace LeftMerge199591
def owner : Owner := ⟨.program ⟨257⟩, ⟨52376⟩⟩
def mergeEvent : Nat := 199591
def frameStart : Nat := 199525
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events779.exact199587RawTerms
def rightRaw : List Term := Proof.Events779.exact199585RawTerms
def group : MergeGroup := .operator 199587 199585
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199587) (leftOrdinal := 0)
    (rightResult := 199585) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199591

namespace LeftMerge199603
def owner : Owner := ⟨.program ⟨257⟩, ⟨53015⟩⟩
def mergeEvent : Nat := 199603
def frameStart : Nat := 199525
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩] } }
def leftRaw : List Term := Proof.Events779.exact199599RawTerms
def rightRaw : List Term := Proof.Events779.exact199576RawTerms
def group : MergeGroup := .operator 199599 199576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199599) (leftOrdinal := 0)
    (rightResult := 199576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53014⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199603

namespace LeftMerge199604
def owner : Owner := ⟨.program ⟨257⟩, ⟨53015⟩⟩
def mergeEvent : Nat := 199604
def frameStart : Nat := 199525
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩] } }
def leftRaw : List Term := Proof.Events779.exact199599RawTerms
def rightRaw : List Term := Proof.Events779.exact199576RawTerms
def group : MergeGroup := .operator 199599 199576
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199599) (leftOrdinal := 1)
    (rightResult := 199576) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53014⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199604

namespace LeftMerge199606
def owner : Owner := ⟨.program ⟨257⟩, ⟨53015⟩⟩
def mergeEvent : Nat := 199606
def frameStart : Nat := 199525
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50904⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52179⟩⟩] } }
def rhsRaw : List Term := Proof.Events779.exact199573RawTerms
def group : MergeGroup := .relation 199605
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199605) (rhsResult := 199573)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53014⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53014⟩⟩) ⟨52179⟩ 199573) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52179⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50904⟩⟩], [⟨.program ⟨257⟩, ⟨52179⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge199606

namespace LeftMerge199614
def owner : Owner := ⟨.program ⟨257⟩, ⟨51201⟩⟩
def mergeEvent : Nat := 199614
def frameStart : Nat := 199525
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51199⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events779.exact199587RawTerms
def rightRaw : List Term := Proof.Events779.exact199610RawTerms
def group : MergeGroup := .operator 199587 199610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 199587) (leftOrdinal := 0)
    (rightResult := 199610) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51199⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51199⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199614

namespace LeftMerge199631
def owner : Owner := ⟨.program ⟨257⟩, ⟨51799⟩⟩
def mergeEvent : Nat := 199631
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }
def rhsRaw : List Term := Proof.Events779.exact199628RawTerms
def group : MergeGroup := .relation 199630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 199630) (rhsResult := 199628)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 199629 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51796⟩⟩]⟩) (none) 199628) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7206⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge199631

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
