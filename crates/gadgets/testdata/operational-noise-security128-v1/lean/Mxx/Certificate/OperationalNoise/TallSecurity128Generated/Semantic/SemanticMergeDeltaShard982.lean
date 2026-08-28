import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge161426
def owner : Owner := ⟨.program ⟨257⟩, ⟨63615⟩⟩
def mergeEvent : Nat := 161426
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨62784⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64053⟩⟩] } }
def rhsRaw : List Term := Proof.Events630.exact161421RawTerms
def group : MergeGroup := .relation 161423
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 161423) (rhsResult := 161421)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63612⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 161422 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63612⟩⟩]⟩) (none) 161421) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62784⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64053⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨62784⟩⟩], [⟨.program ⟨257⟩, ⟨64053⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge161426

namespace LeftMerge161427
def owner : Owner := ⟨.program ⟨257⟩, ⟨63615⟩⟩
def mergeEvent : Nat := 161427
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events630.exact161421RawTerms
def group : MergeGroup := .relation 161423
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 161423) (rhsResult := 161421)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63612⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 161422 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63612⟩⟩]⟩) (none) 161421) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge161427

namespace LeftMerge161432
def owner : Owner := ⟨.program ⟨257⟩, ⟨64775⟩⟩
def mergeEvent : Nat := 161432
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64772⟩⟩] } }
def leftRaw : List Term := Proof.Events630.exact161428RawTerms
def rightRaw : List Term := Proof.Events629.exact161250RawTerms
def group : MergeGroup := .operator 161428 161250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 161428) (leftOrdinal := 0)
    (rightResult := 161250) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64772⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64772⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨64772⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge161432

namespace LeftMerge161433
def owner : Owner := ⟨.program ⟨257⟩, ⟨64775⟩⟩
def mergeEvent : Nat := 161433
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨62784⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64053⟩⟩] } }
def leftRaw : List Term := Proof.Events630.exact161428RawTerms
def rightRaw : List Term := Proof.Events629.exact161250RawTerms
def group : MergeGroup := .operator 161428 161250
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 161428) (leftOrdinal := 2)
    (rightResult := 161250) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨62784⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64053⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨62784⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64053⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨62784⟩⟩], [⟨.program ⟨257⟩, ⟨64053⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge161433

namespace LeftMerge161441
def owner : Owner := ⟨.program ⟨257⟩, ⟨64776⟩⟩
def mergeEvent : Nat := 161441
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩] } }
def leftRaw : List Term := Proof.Events630.exact161435RawTerms
def rightRaw : List Term := Proof.Events061.exact15722RawTerms
def group : MergeGroup := .operator 161435 15722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 161435) (leftOrdinal := 0)
    (rightResult := 15722) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7213⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7099⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7213⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge161441

namespace LeftMerge161442
def owner : Owner := ⟨.program ⟨257⟩, ⟨64776⟩⟩
def mergeEvent : Nat := 161442
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩] } }
def leftRaw : List Term := Proof.Events630.exact161435RawTerms
def rightRaw : List Term := Proof.Events061.exact15722RawTerms
def group : MergeGroup := .operator 161435 15722
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 161435) (leftOrdinal := 1)
    (rightResult := 15722) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7099⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge161442

namespace LeftMerge161444
def owner : Owner := ⟨.program ⟨257⟩, ⟨64776⟩⟩
def mergeEvent : Nat := 161444
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15715RawTerms
def group : MergeGroup := .relation 161443
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 161443) (rhsResult := 15715)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7099⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7099⟩⟩) ⟨7015⟩ 15715) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63028⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge161444

namespace LeftMerge161458
def owner : Owner := ⟨.program ⟨257⟩, ⟨61794⟩⟩
def mergeEvent : Nat := 161458
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩] } }
def leftRaw : List Term := Proof.Events602.exact154126RawTerms
def rightRaw : List Term := Proof.Events630.exact161452RawTerms
def group : MergeGroup := .operator 154126 161452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154126) (leftOrdinal := 0)
    (rightResult := 161452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61792⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge161458

namespace LeftMerge161459
def owner : Owner := ⟨.program ⟨257⟩, ⟨61794⟩⟩
def mergeEvent : Nat := 161459
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩] } }
def leftRaw : List Term := Proof.Events602.exact154126RawTerms
def rightRaw : List Term := Proof.Events630.exact161452RawTerms
def group : MergeGroup := .operator 154126 161452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 154126) (leftOrdinal := 1)
    (rightResult := 161452) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61792⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge161459

namespace LeftMerge161461
def owner : Owner := ⟨.program ⟨257⟩, ⟨61794⟩⟩
def mergeEvent : Nat := 161461
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61073⟩⟩] } }
def rhsRaw : List Term := Proof.Events630.exact161449RawTerms
def group : MergeGroup := .relation 161460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 161460) (rhsResult := 161449)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61792⟩⟩) ⟨61073⟩ 161449) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61073⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61073⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge161461

namespace LeftMerge161475
def owner : Owner := ⟨.program ⟨257⟩, ⟨60635⟩⟩
def mergeEvent : Nat := 161475
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60632⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events630.exact161469RawTerms
def group : MergeGroup := .operator 149120 161469
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 161469) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨60632⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨60632⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge161475

namespace LeftMerge161596
def owner : Owner := ⟨.program ⟨257⟩, ⟨61296⟩⟩
def mergeEvent : Nat := 161596
def frameStart : Nat := 161530
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events631.exact161592RawTerms
def rightRaw : List Term := Proof.Events631.exact161590RawTerms
def group : MergeGroup := .operator 161592 161590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 161592) (leftOrdinal := 0)
    (rightResult := 161590) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge161596

namespace LeftMerge161608
def owner : Owner := ⟨.program ⟨257⟩, ⟨61793⟩⟩
def mergeEvent : Nat := 161608
def frameStart : Nat := 161530
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩] } }
def leftRaw : List Term := Proof.Events631.exact161604RawTerms
def rightRaw : List Term := Proof.Events631.exact161581RawTerms
def group : MergeGroup := .operator 161604 161581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 161604) (leftOrdinal := 0)
    (rightResult := 161581) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7186⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61792⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7186⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge161608

namespace LeftMerge161609
def owner : Owner := ⟨.program ⟨257⟩, ⟨61793⟩⟩
def mergeEvent : Nat := 161609
def frameStart : Nat := 161530
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩] } }
def leftRaw : List Term := Proof.Events631.exact161604RawTerms
def rightRaw : List Term := Proof.Events631.exact161581RawTerms
def group : MergeGroup := .operator 161604 161581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 161604) (leftOrdinal := 1)
    (rightResult := 161581) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61792⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge161609

namespace LeftMerge161611
def owner : Owner := ⟨.program ⟨257⟩, ⟨61793⟩⟩
def mergeEvent : Nat := 161611
def frameStart : Nat := 161530
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨59804⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨61073⟩⟩] } }
def rhsRaw : List Term := Proof.Events631.exact161578RawTerms
def group : MergeGroup := .relation 161610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 161610) (rhsResult := 161578)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨61792⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨61792⟩⟩) ⟨61073⟩ 161578) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨61073⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨59804⟩⟩], [⟨.program ⟨257⟩, ⟨61073⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge161611

namespace LeftMerge161619
def owner : Owner := ⟨.program ⟨257⟩, ⟨60051⟩⟩
def mergeEvent : Nat := 161619
def frameStart : Nat := 161530
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events631.exact161592RawTerms
def rightRaw : List Term := Proof.Events631.exact161615RawTerms
def group : MergeGroup := .operator 161592 161615
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 161592) (leftOrdinal := 0)
    (rightResult := 161615) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60048⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge161619

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
