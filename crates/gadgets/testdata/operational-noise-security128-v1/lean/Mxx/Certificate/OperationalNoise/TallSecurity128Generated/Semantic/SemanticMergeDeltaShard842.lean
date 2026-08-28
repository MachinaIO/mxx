import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge138543
def owner : Owner := ⟨.program ⟨257⟩, ⟨69626⟩⟩
def mergeEvent : Nat := 138543
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138537RawTerms
def rightRaw : List Term := Proof.Events540.exact138260RawTerms
def group : MergeGroup := .operator 138537 138260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138537) (leftOrdinal := 0)
    (rightResult := 138260) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69624⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138543

namespace LeftMerge138544
def owner : Owner := ⟨.program ⟨257⟩, ⟨69626⟩⟩
def mergeEvent : Nat := 138544
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138537RawTerms
def rightRaw : List Term := Proof.Events540.exact138260RawTerms
def group : MergeGroup := .operator 138537 138260
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138537) (leftOrdinal := 1)
    (rightResult := 138260) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69624⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138544

namespace LeftMerge138546
def owner : Owner := ⟨.program ⟨257⟩, ⟨69626⟩⟩
def mergeEvent : Nat := 138546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68619⟩⟩] } }
def rhsRaw : List Term := Proof.Events540.exact138257RawTerms
def group : MergeGroup := .relation 138545
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138545) (rhsResult := 138257)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69624⟩⟩) ⟨68619⟩ 138257) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68619⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨68619⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138546

namespace LeftMerge138560
def owner : Owner := ⟨.program ⟨257⟩, ⟨67940⟩⟩
def mergeEvent : Nat := 138560
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩] } }
def leftRaw : List Term := Proof.Events525.exact134495RawTerms
def rightRaw : List Term := Proof.Events541.exact138554RawTerms
def group : MergeGroup := .operator 134495 138554
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 134495) (leftOrdinal := 0)
    (rightResult := 138554) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨67937⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138560

namespace LeftMerge138681
def owner : Owner := ⟨.program ⟨257⟩, ⟨68981⟩⟩
def mergeEvent : Nat := 138681
def frameStart : Nat := 138615
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138677RawTerms
def rightRaw : List Term := Proof.Events541.exact138675RawTerms
def group : MergeGroup := .operator 138677 138675
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138677) (leftOrdinal := 0)
    (rightResult := 138675) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138681

namespace LeftMerge138693
def owner : Owner := ⟨.program ⟨257⟩, ⟨69625⟩⟩
def mergeEvent : Nat := 138693
def frameStart : Nat := 138615
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138689RawTerms
def rightRaw : List Term := Proof.Events541.exact138666RawTerms
def group : MergeGroup := .operator 138689 138666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138689) (leftOrdinal := 0)
    (rightResult := 138666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69624⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138693

namespace LeftMerge138694
def owner : Owner := ⟨.program ⟨257⟩, ⟨69625⟩⟩
def mergeEvent : Nat := 138694
def frameStart : Nat := 138615
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138689RawTerms
def rightRaw : List Term := Proof.Events541.exact138666RawTerms
def group : MergeGroup := .operator 138689 138666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138689) (leftOrdinal := 1)
    (rightResult := 138666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨69624⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138694

namespace LeftMerge138696
def owner : Owner := ⟨.program ⟨257⟩, ⟨69625⟩⟩
def mergeEvent : Nat := 138696
def frameStart : Nat := 138615
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68619⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138663RawTerms
def group : MergeGroup := .relation 138695
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138695) (rhsResult := 138663)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨69624⟩⟩) ⟨68619⟩ 138663) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68619⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨68619⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138696

namespace LeftMerge138704
def owner : Owner := ⟨.program ⟨257⟩, ⟨66122⟩⟩
def mergeEvent : Nat := 138704
def frameStart : Nat := 138615
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138677RawTerms
def rightRaw : List Term := Proof.Events541.exact138700RawTerms
def group : MergeGroup := .operator 138677 138700
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138677) (leftOrdinal := 0)
    (rightResult := 138700) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138704

namespace LeftMerge138721
def owner : Owner := ⟨.program ⟨257⟩, ⟨67940⟩⟩
def mergeEvent : Nat := 138721
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138718RawTerms
def group : MergeGroup := .relation 138720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138720) (rhsResult := 138718)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 138719 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩]⟩) (none) 138718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7216⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138721

namespace LeftMerge138722
def owner : Owner := ⟨.program ⟨257⟩, ⟨67940⟩⟩
def mergeEvent : Nat := 138722
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138718RawTerms
def group : MergeGroup := .relation 138720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138720) (rhsResult := 138718)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 138719 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩]⟩) (none) 138718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138722

namespace LeftMerge138723
def owner : Owner := ⟨.program ⟨257⟩, ⟨67940⟩⟩
def mergeEvent : Nat := 138723
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68619⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138718RawTerms
def group : MergeGroup := .relation 138720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138720) (rhsResult := 138718)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 138719 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩]⟩) (none) 138718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68619⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨68619⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138723

namespace LeftMerge138724
def owner : Owner := ⟨.program ⟨257⟩, ⟨67940⟩⟩
def mergeEvent : Nat := 138724
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events541.exact138718RawTerms
def group : MergeGroup := .relation 138720
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 138720) (rhsResult := 138718)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 138719 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨67937⟩⟩]⟩) (none) 138718) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66111⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨66111⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138724

namespace LeftMerge138729
def owner : Owner := ⟨.program ⟨257⟩, ⟨69627⟩⟩
def mergeEvent : Nat := 138729
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138725RawTerms
def rightRaw : List Term := Proof.Events541.exact138547RawTerms
def group : MergeGroup := .operator 138725 138547
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138725) (leftOrdinal := 0)
    (rightResult := 138547) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩], [⟨.program ⟨257⟩, ⟨7188⟩⟩, ⟨.program ⟨257⟩, ⟨69624⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138729

namespace LeftMerge138730
def owner : Owner := ⟨.program ⟨257⟩, ⟨69627⟩⟩
def mergeEvent : Nat := 138730
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68619⟩⟩] } }
def leftRaw : List Term := Proof.Events541.exact138725RawTerms
def rightRaw : List Term := Proof.Events541.exact138547RawTerms
def group : MergeGroup := .operator 138725 138547
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 138725) (leftOrdinal := 2)
    (rightResult := 138547) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68619⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68619⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨65732⟩⟩], [⟨.program ⟨257⟩, ⟨68619⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge138730

namespace LeftMerge138756
def owner : Owner := ⟨.program ⟨257⟩, ⟨25407⟩⟩
def mergeEvent : Nat := 138756
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events024.exact6285RawTerms
def rightRaw : List Term := Proof.Events525.exact134403RawTerms
def group : MergeGroup := .operator 6285 134403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 6285) (leftOrdinal := 0)
    (rightResult := 134403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25406⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2945⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2945⟩⟩, ⟨.program ⟨257⟩, ⟨25406⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge138756

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
