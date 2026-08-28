import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge36450
def owner : Owner := ⟨.program ⟨257⟩, ⟨64539⟩⟩
def mergeEvent : Nat := 36450
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36441RawTerms
def rightRaw : List Term := Proof.Events142.exact36377RawTerms
def group : MergeGroup := .operator 36441 36377
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36441) (leftOrdinal := 0)
    (rightResult := 36377) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36450

namespace LeftMerge36464
def owner : Owner := ⟨.program ⟨257⟩, ⟨63462⟩⟩
def mergeEvent : Nat := 36464
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events142.exact36458RawTerms
def group : MergeGroup := .operator 32120 36458
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 36458) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63459⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36464

namespace LeftMerge36543
def owner : Owner := ⟨.program ⟨257⟩, ⟨62709⟩⟩
def mergeEvent : Nat := 36543
def frameStart : Nat := 36513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events142.exact36539RawTerms
def rightRaw : List Term := Proof.Events142.exact36536RawTerms
def group : MergeGroup := .operator 36539 36536
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36539) (leftOrdinal := 0)
    (rightResult := 36536) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25598⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36543

namespace LeftMerge36573
def owner : Owner := ⟨.program ⟨257⟩, ⟨64244⟩⟩
def mergeEvent : Nat := 36573
def frameStart : Nat := 36513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36569RawTerms
def rightRaw : List Term := Proof.Events142.exact36567RawTerms
def group : MergeGroup := .operator 36569 36567
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36569) (leftOrdinal := 0)
    (rightResult := 36567) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36573

namespace LeftMerge36596
def owner : Owner := ⟨.program ⟨257⟩, ⟨9540⟩⟩
def mergeEvent : Nat := 36596
def frameStart : Nat := 36513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36592RawTerms
def rightRaw : List Term := Proof.Events142.exact36589RawTerms
def group : MergeGroup := .operator 36592 36589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36592) (leftOrdinal := 0)
    (rightResult := 36589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36596

namespace LeftMerge36605
def owner : Owner := ⟨.program ⟨257⟩, ⟨64541⟩⟩
def mergeEvent : Nat := 36605
def frameStart : Nat := 36513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36601RawTerms
def rightRaw : List Term := Proof.Events142.exact36558RawTerms
def group : MergeGroup := .operator 36601 36558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36601) (leftOrdinal := 0)
    (rightResult := 36558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64538⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36605

namespace LeftMerge36606
def owner : Owner := ⟨.program ⟨257⟩, ⟨64541⟩⟩
def mergeEvent : Nat := 36606
def frameStart : Nat := 36513
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36601RawTerms
def rightRaw : List Term := Proof.Events142.exact36558RawTerms
def group : MergeGroup := .operator 36601 36558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36601) (leftOrdinal := 1)
    (rightResult := 36558) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36606

namespace LeftMerge36608
def owner : Owner := ⟨.program ⟨257⟩, ⟨64541⟩⟩
def mergeEvent : Nat := 36608
def frameStart : Nat := 36513
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63983⟩⟩] } }
def rhsRaw : List Term := Proof.Events142.exact36555RawTerms
def group : MergeGroup := .relation 36607
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36607) (rhsResult := 36555)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨64538⟩⟩) ⟨63983⟩ 36555) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨63983⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], [⟨.program ⟨257⟩, ⟨63983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36608

namespace LeftMerge36616
def owner : Owner := ⟨.program ⟨257⟩, ⟨62882⟩⟩
def mergeEvent : Nat := 36616
def frameStart : Nat := 36513
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events142.exact36569RawTerms
def rightRaw : List Term := Proof.Events143.exact36612RawTerms
def group : MergeGroup := .operator 36569 36612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36569) (leftOrdinal := 0)
    (rightResult := 36612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36616

namespace LeftMerge36633
def owner : Owner := ⟨.program ⟨257⟩, ⟨63462⟩⟩
def mergeEvent : Nat := 36633
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36630RawTerms
def group : MergeGroup := .relation 36632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36632) (rhsResult := 36630)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36631 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩]⟩) (none) 36630) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36633

namespace LeftMerge36634
def owner : Owner := ⟨.program ⟨257⟩, ⟨63462⟩⟩
def mergeEvent : Nat := 36634
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36630RawTerms
def group : MergeGroup := .relation 36632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36632) (rhsResult := 36630)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36631 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩]⟩) (none) 36630) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36634

namespace LeftMerge36635
def owner : Owner := ⟨.program ⟨257⟩, ⟨63462⟩⟩
def mergeEvent : Nat := 36635
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63983⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36630RawTerms
def group : MergeGroup := .relation 36632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36632) (rhsResult := 36630)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36631 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩]⟩) (none) 36630) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63983⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], [⟨.program ⟨257⟩, ⟨63983⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36635

namespace LeftMerge36636
def owner : Owner := ⟨.program ⟨257⟩, ⟨63462⟩⟩
def mergeEvent : Nat := 36636
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events143.exact36630RawTerms
def group : MergeGroup := .relation 36632
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 36632) (rhsResult := 36630)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 36631 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63459⟩⟩]⟩) (none) 36630) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62880⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨62880⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36636

namespace LeftMerge36641
def owner : Owner := ⟨.program ⟨257⟩, ⟨64540⟩⟩
def mergeEvent : Nat := 36641
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63983⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36637RawTerms
def rightRaw : List Term := Proof.Events142.exact36451RawTerms
def group : MergeGroup := .operator 36637 36451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36637) (leftOrdinal := 2)
    (rightResult := 36451) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63983⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨63983⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨25598⟩⟩, ⟨.program ⟨257⟩, ⟨62708⟩⟩], [⟨.program ⟨257⟩, ⟨63983⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge36641

namespace LeftMerge36642
def owner : Owner := ⟨.program ⟨257⟩, ⟨64540⟩⟩
def mergeEvent : Nat := 36642
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36637RawTerms
def rightRaw : List Term := Proof.Events142.exact36451RawTerms
def group : MergeGroup := .operator 36637 36451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36637) (leftOrdinal := 1)
    (rightResult := 36451) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7293⟩⟩, ⟨.program ⟨257⟩, ⟨9538⟩⟩, ⟨.program ⟨257⟩, ⟨64538⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36642

namespace LeftMerge36650
def owner : Owner := ⟨.program ⟨257⟩, ⟨65153⟩⟩
def mergeEvent : Nat := 36650
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36644RawTerms
def rightRaw : List Term := Proof.Events142.exact36367RawTerms
def group : MergeGroup := .operator 36644 36367
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36644) (leftOrdinal := 0)
    (rightResult := 36367) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65151⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65151⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge36650

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
