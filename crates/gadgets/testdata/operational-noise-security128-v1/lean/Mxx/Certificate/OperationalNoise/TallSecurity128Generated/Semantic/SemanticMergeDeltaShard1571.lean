import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge254564
def owner : Owner := ⟨.program ⟨257⟩, ⟨29482⟩⟩
def mergeEvent : Nat := 254564
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30059⟩⟩] } }
def rhsRaw : List Term := Proof.Events994.exact254559RawTerms
def group : MergeGroup := .relation 254561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254561) (rhsResult := 254559)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 254560 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29479⟩⟩]⟩) (none) 254559) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13206⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30059⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], [⟨.program ⟨257⟩, ⟨30059⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254564

namespace LeftMerge254565
def owner : Owner := ⟨.program ⟨257⟩, ⟨29482⟩⟩
def mergeEvent : Nat := 254565
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events994.exact254559RawTerms
def group : MergeGroup := .relation 254561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254561) (rhsResult := 254559)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29479⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 254560 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29479⟩⟩]⟩) (none) 254559) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254565

namespace LeftMerge254570
def owner : Owner := ⟨.program ⟨257⟩, ⟨30546⟩⟩
def mergeEvent : Nat := 254570
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30059⟩⟩] } }
def leftRaw : List Term := Proof.Events994.exact254566RawTerms
def rightRaw : List Term := Proof.Events993.exact254380RawTerms
def group : MergeGroup := .operator 254566 254380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254566) (leftOrdinal := 2)
    (rightResult := 254380) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30059⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30059⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨13206⟩⟩, ⟨.program ⟨257⟩, ⟨28654⟩⟩], [⟨.program ⟨257⟩, ⟨30059⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254570

namespace LeftMerge254571
def owner : Owner := ⟨.program ⟨257⟩, ⟨30546⟩⟩
def mergeEvent : Nat := 254571
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30544⟩⟩] } }
def leftRaw : List Term := Proof.Events994.exact254566RawTerms
def rightRaw : List Term := Proof.Events993.exact254380RawTerms
def group : MergeGroup := .operator 254566 254380
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254566) (leftOrdinal := 1)
    (rightResult := 254380) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30544⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254571

namespace LeftMerge254579
def owner : Owner := ⟨.program ⟨257⟩, ⟨30846⟩⟩
def mergeEvent : Nat := 254579
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩] } }
def leftRaw : List Term := Proof.Events994.exact254573RawTerms
def rightRaw : List Term := Proof.Events993.exact254296RawTerms
def group : MergeGroup := .operator 254573 254296
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254573) (leftOrdinal := 0)
    (rightResult := 254296) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30844⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254579

namespace LeftMerge254580
def owner : Owner := ⟨.program ⟨257⟩, ⟨30846⟩⟩
def mergeEvent : Nat := 254580
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩] } }
def leftRaw : List Term := Proof.Events994.exact254573RawTerms
def rightRaw : List Term := Proof.Events993.exact254296RawTerms
def group : MergeGroup := .operator 254573 254296
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254573) (leftOrdinal := 1)
    (rightResult := 254296) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30844⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254580

namespace LeftMerge254582
def owner : Owner := ⟨.program ⟨257⟩, ⟨30846⟩⟩
def mergeEvent : Nat := 254582
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30196⟩⟩] } }
def rhsRaw : List Term := Proof.Events993.exact254293RawTerms
def group : MergeGroup := .relation 254581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254581) (rhsResult := 254293)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30844⟩⟩) ⟨30196⟩ 254293) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], [⟨.program ⟨257⟩, ⟨30196⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254582

namespace LeftMerge254596
def owner : Owner := ⟨.program ⟨257⟩, ⟨29739⟩⟩
def mergeEvent : Nat := 254596
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29736⟩⟩] } }
def leftRaw : List Term := Proof.Events982.exact251495RawTerms
def rightRaw : List Term := Proof.Events994.exact254590RawTerms
def group : MergeGroup := .operator 251495 254590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 251495) (leftOrdinal := 0)
    (rightResult := 254590) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29736⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29736⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254596

namespace LeftMerge254717
def owner : Owner := ⟨.program ⟨257⟩, ⟨30428⟩⟩
def mergeEvent : Nat := 254717
def frameStart : Nat := 254651
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events994.exact254713RawTerms
def rightRaw : List Term := Proof.Events994.exact254711RawTerms
def group : MergeGroup := .operator 254713 254711
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254713) (leftOrdinal := 0)
    (rightResult := 254711) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254717

namespace LeftMerge254729
def owner : Owner := ⟨.program ⟨257⟩, ⟨30845⟩⟩
def mergeEvent : Nat := 254729
def frameStart : Nat := 254651
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩] } }
def leftRaw : List Term := Proof.Events995.exact254725RawTerms
def rightRaw : List Term := Proof.Events994.exact254702RawTerms
def group : MergeGroup := .operator 254725 254702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254725) (leftOrdinal := 0)
    (rightResult := 254702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30844⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254729

namespace LeftMerge254730
def owner : Owner := ⟨.program ⟨257⟩, ⟨30845⟩⟩
def mergeEvent : Nat := 254730
def frameStart : Nat := 254651
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩] } }
def leftRaw : List Term := Proof.Events995.exact254725RawTerms
def rightRaw : List Term := Proof.Events994.exact254702RawTerms
def group : MergeGroup := .operator 254725 254702
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254725) (leftOrdinal := 1)
    (rightResult := 254702) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30844⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254730

namespace LeftMerge254732
def owner : Owner := ⟨.program ⟨257⟩, ⟨30845⟩⟩
def mergeEvent : Nat := 254732
def frameStart : Nat := 254651
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30196⟩⟩] } }
def rhsRaw : List Term := Proof.Events994.exact254699RawTerms
def group : MergeGroup := .relation 254731
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254731) (rhsResult := 254699)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29048⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30844⟩⟩) ⟨30196⟩ 254699) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29048⟩⟩], [⟨.program ⟨257⟩, ⟨30196⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254732

namespace LeftMerge254740
def owner : Owner := ⟨.program ⟨257⟩, ⟨29235⟩⟩
def mergeEvent : Nat := 254740
def frameStart : Nat := 254651
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events994.exact254713RawTerms
def rightRaw : List Term := Proof.Events995.exact254736RawTerms
def group : MergeGroup := .operator 254713 254736
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 254713) (leftOrdinal := 0)
    (rightResult := 254736) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29234⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254740

namespace LeftMerge254757
def owner : Owner := ⟨.program ⟨257⟩, ⟨29739⟩⟩
def mergeEvent : Nat := 254757
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }
def rhsRaw : List Term := Proof.Events995.exact254754RawTerms
def group : MergeGroup := .relation 254756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254756) (rhsResult := 254754)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 254755 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29736⟩⟩]⟩) (none) 254754) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7220⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254757

namespace LeftMerge254758
def owner : Owner := ⟨.program ⟨257⟩, ⟨29739⟩⟩
def mergeEvent : Nat := 254758
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩] } }
def rhsRaw : List Term := Proof.Events995.exact254754RawTerms
def group : MergeGroup := .relation 254756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254756) (rhsResult := 254754)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 254755 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29736⟩⟩]⟩) (none) 254754) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30844⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge254758

namespace LeftMerge254759
def owner : Owner := ⟨.program ⟨257⟩, ⟨29739⟩⟩
def mergeEvent : Nat := 254759
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30196⟩⟩] } }
def rhsRaw : List Term := Proof.Events995.exact254754RawTerms
def group : MergeGroup := .relation 254756
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 254756) (rhsResult := 254754)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29736⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 254755 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29736⟩⟩]⟩) (none) 254754) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29048⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨4743⟩⟩, ⟨.program ⟨257⟩, ⟨29048⟩⟩], [⟨.program ⟨257⟩, ⟨30196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge254759

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
