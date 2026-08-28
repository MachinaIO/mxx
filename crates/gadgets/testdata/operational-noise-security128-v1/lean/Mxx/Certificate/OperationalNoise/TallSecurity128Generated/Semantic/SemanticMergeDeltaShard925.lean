import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge151522
def owner : Owner := ⟨.program ⟨257⟩, ⟨36227⟩⟩
def mergeEvent : Nat := 151522
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩] } }
def leftRaw : List Term := Proof.Events591.exact151513RawTerms
def rightRaw : List Term := Proof.Events591.exact151449RawTerms
def group : MergeGroup := .operator 151513 151449
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151513) (leftOrdinal := 0)
    (rightResult := 151449) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36226⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151522

namespace LeftMerge151536
def owner : Owner := ⟨.program ⟨257⟩, ⟨35162⟩⟩
def mergeEvent : Nat := 151536
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩] } }
def leftRaw : List Term := Proof.Events582.exact149120RawTerms
def rightRaw : List Term := Proof.Events591.exact151530RawTerms
def group : MergeGroup := .operator 149120 151530
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 149120) (leftOrdinal := 0)
    (rightResult := 151530) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35159⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151536

namespace LeftMerge151615
def owner : Owner := ⟨.program ⟨257⟩, ⟨34363⟩⟩
def mergeEvent : Nat := 151615
def frameStart : Nat := 151585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events592.exact151611RawTerms
def rightRaw : List Term := Proof.Events592.exact151608RawTerms
def group : MergeGroup := .operator 151611 151608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151611) (leftOrdinal := 0)
    (rightResult := 151608) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13536⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151615

namespace LeftMerge151645
def owner : Owner := ⟨.program ⟨257⟩, ⟨36016⟩⟩
def mergeEvent : Nat := 151645
def frameStart : Nat := 151585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events592.exact151641RawTerms
def rightRaw : List Term := Proof.Events592.exact151639RawTerms
def group : MergeGroup := .operator 151641 151639
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151641) (leftOrdinal := 0)
    (rightResult := 151639) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151645

namespace LeftMerge151668
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def mergeEvent : Nat := 151668
def frameStart : Nat := 151585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }
def leftRaw : List Term := Proof.Events592.exact151664RawTerms
def rightRaw : List Term := Proof.Events592.exact151661RawTerms
def group : MergeGroup := .operator 151664 151661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151664) (leftOrdinal := 0)
    (rightResult := 151661) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9550⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151668

namespace LeftMerge151677
def owner : Owner := ⟨.program ⟨257⟩, ⟨36229⟩⟩
def mergeEvent : Nat := 151677
def frameStart : Nat := 151585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩] } }
def leftRaw : List Term := Proof.Events592.exact151673RawTerms
def rightRaw : List Term := Proof.Events592.exact151630RawTerms
def group : MergeGroup := .operator 151673 151630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151673) (leftOrdinal := 0)
    (rightResult := 151630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36226⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151677

namespace LeftMerge151678
def owner : Owner := ⟨.program ⟨257⟩, ⟨36229⟩⟩
def mergeEvent : Nat := 151678
def frameStart : Nat := 151585
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩] } }
def leftRaw : List Term := Proof.Events592.exact151673RawTerms
def rightRaw : List Term := Proof.Events592.exact151630RawTerms
def group : MergeGroup := .operator 151673 151630
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151673) (leftOrdinal := 1)
    (rightResult := 151630) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36226⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151678

namespace LeftMerge151680
def owner : Owner := ⟨.program ⟨257⟩, ⟨36229⟩⟩
def mergeEvent : Nat := 151680
def frameStart : Nat := 151585
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35731⟩⟩] } }
def rhsRaw : List Term := Proof.Events592.exact151627RawTerms
def group : MergeGroup := .relation 151679
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151679) (rhsResult := 151627)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36226⟩⟩) ⟨35731⟩ 151627) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35731⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], [⟨.program ⟨257⟩, ⟨35731⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151680

namespace LeftMerge151688
def owner : Owner := ⟨.program ⟨257⟩, ⟨34726⟩⟩
def mergeEvent : Nat := 151688
def frameStart : Nat := 151585
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34724⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events592.exact151641RawTerms
def rightRaw : List Term := Proof.Events592.exact151684RawTerms
def group : MergeGroup := .operator 151641 151684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151641) (leftOrdinal := 0)
    (rightResult := 151684) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34724⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34724⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151688

namespace LeftMerge151705
def owner : Owner := ⟨.program ⟨257⟩, ⟨35162⟩⟩
def mergeEvent : Nat := 151705
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }
def rhsRaw : List Term := Proof.Events592.exact151702RawTerms
def group : MergeGroup := .relation 151704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151704) (rhsResult := 151702)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 151703 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩]⟩) (none) 151702) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151705

namespace LeftMerge151706
def owner : Owner := ⟨.program ⟨257⟩, ⟨35162⟩⟩
def mergeEvent : Nat := 151706
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩] } }
def rhsRaw : List Term := Proof.Events592.exact151702RawTerms
def group : MergeGroup := .relation 151704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151704) (rhsResult := 151702)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 151703 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩]⟩) (none) 151702) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151706

namespace LeftMerge151707
def owner : Owner := ⟨.program ⟨257⟩, ⟨35162⟩⟩
def mergeEvent : Nat := 151707
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35731⟩⟩] } }
def rhsRaw : List Term := Proof.Events592.exact151702RawTerms
def group : MergeGroup := .relation 151704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151704) (rhsResult := 151702)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 151703 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩]⟩) (none) 151702) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35731⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], [⟨.program ⟨257⟩, ⟨35731⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151707

namespace LeftMerge151708
def owner : Owner := ⟨.program ⟨257⟩, ⟨35162⟩⟩
def mergeEvent : Nat := 151708
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34724⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events592.exact151702RawTerms
def group : MergeGroup := .relation 151704
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 151704) (rhsResult := 151702)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 151703 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35159⟩⟩]⟩) (none) 151702) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34724⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨34724⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151708

namespace LeftMerge151713
def owner : Owner := ⟨.program ⟨257⟩, ⟨36228⟩⟩
def mergeEvent : Nat := 151713
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35731⟩⟩] } }
def leftRaw : List Term := Proof.Events592.exact151709RawTerms
def rightRaw : List Term := Proof.Events591.exact151523RawTerms
def group : MergeGroup := .operator 151709 151523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151709) (leftOrdinal := 2)
    (rightResult := 151523) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35731⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35731⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩, ⟨.program ⟨257⟩, ⟨13536⟩⟩, ⟨.program ⟨257⟩, ⟨34362⟩⟩], [⟨.program ⟨257⟩, ⟨35731⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge151713

namespace LeftMerge151714
def owner : Owner := ⟨.program ⟨257⟩, ⟨36228⟩⟩
def mergeEvent : Nat := 151714
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩] } }
def leftRaw : List Term := Proof.Events592.exact151709RawTerms
def rightRaw : List Term := Proof.Events591.exact151523RawTerms
def group : MergeGroup := .operator 151709 151523
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151709) (leftOrdinal := 1)
    (rightResult := 151523) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7297⟩⟩, ⟨.program ⟨257⟩, ⟨9550⟩⟩, ⟨.program ⟨257⟩, ⟨36226⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151714

namespace LeftMerge151722
def owner : Owner := ⟨.program ⟨257⟩, ⟨36556⟩⟩
def mergeEvent : Nat := 151722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36554⟩⟩] } }
def leftRaw : List Term := Proof.Events592.exact151716RawTerms
def rightRaw : List Term := Proof.Events591.exact151439RawTerms
def group : MergeGroup := .operator 151716 151439
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 151716) (leftOrdinal := 0)
    (rightResult := 151439) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4617⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36554⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4617⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36554⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge151722

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
