import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge91440
def owner : Owner := ⟨.program ⟨214⟩, ⟨28947⟩⟩
def mergeEvent : Nat := 91440
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28944⟩⟩] } }
def leftRaw : List Term := Proof.Events357.exact91436RawTerms
def rightRaw : List Term := Proof.Events356.exact91258RawTerms
def group : MergeGroup := .operator 91436 91258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91436) (leftOrdinal := 0)
    (rightResult := 91258) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28944⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28944⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6702⟩⟩, ⟨.program ⟨214⟩, ⟨28944⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91440

namespace LeftMerge91441
def owner : Owner := ⟨.program ⟨214⟩, ⟨28947⟩⟩
def mergeEvent : Nat := 91441
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16465⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24476⟩⟩] } }
def leftRaw : List Term := Proof.Events357.exact91436RawTerms
def rightRaw : List Term := Proof.Events356.exact91258RawTerms
def group : MergeGroup := .operator 91436 91258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91436) (leftOrdinal := 2)
    (rightResult := 91258) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16465⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24476⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16465⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24476⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16465⟩⟩], [⟨.program ⟨214⟩, ⟨24476⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91441

namespace LeftMerge91449
def owner : Owner := ⟨.program ⟨214⟩, ⟨28948⟩⟩
def mergeEvent : Nat := 91449
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }
def leftRaw : List Term := Proof.Events357.exact91443RawTerms
def rightRaw : List Term := Proof.Events021.exact5619RawTerms
def group : MergeGroup := .operator 91443 5619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91443) (leftOrdinal := 0)
    (rightResult := 5619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6732⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6669⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6732⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91449

namespace LeftMerge91450
def owner : Owner := ⟨.program ⟨214⟩, ⟨28948⟩⟩
def mergeEvent : Nat := 91450
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩] } }
def leftRaw : List Term := Proof.Events357.exact91443RawTerms
def rightRaw : List Term := Proof.Events021.exact5619RawTerms
def group : MergeGroup := .operator 91443 5619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91443) (leftOrdinal := 1)
    (rightResult := 5619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6669⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91450

namespace LeftMerge91452
def owner : Owner := ⟨.program ⟨214⟩, ⟨28948⟩⟩
def mergeEvent : Nat := 91452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events021.exact5612RawTerms
def group : MergeGroup := .relation 91451
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91451) (rhsResult := 5612)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6669⟩⟩) ⟨6606⟩ 5612) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6473⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17550⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91452

namespace LeftMerge91466
def owner : Owner := ⟨.program ⟨214⟩, ⟨28729⟩⟩
def mergeEvent : Nat := 91466
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩] } }
def leftRaw : List Term := Proof.Events324.exact83076RawTerms
def rightRaw : List Term := Proof.Events357.exact91460RawTerms
def group : MergeGroup := .operator 83076 91460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83076) (leftOrdinal := 0)
    (rightResult := 91460) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28727⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91466

namespace LeftMerge91467
def owner : Owner := ⟨.program ⟨214⟩, ⟨28729⟩⟩
def mergeEvent : Nat := 91467
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩] } }
def leftRaw : List Term := Proof.Events324.exact83076RawTerms
def rightRaw : List Term := Proof.Events357.exact91460RawTerms
def group : MergeGroup := .operator 83076 91460
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83076) (leftOrdinal := 1)
    (rightResult := 91460) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28727⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91467

namespace LeftMerge91469
def owner : Owner := ⟨.program ⟨214⟩, ⟨28729⟩⟩
def mergeEvent : Nat := 91469
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24413⟩⟩] } }
def rhsRaw : List Term := Proof.Events357.exact91457RawTerms
def group : MergeGroup := .relation 91468
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91468) (rhsResult := 91457)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28727⟩⟩) ⟨24413⟩ 91457) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24413⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩, ⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24413⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91469

namespace LeftMerge91483
def owner : Owner := ⟨.program ⟨214⟩, ⟨21907⟩⟩
def mergeEvent : Nat := 91483
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21904⟩⟩] } }
def leftRaw : List Term := Proof.Events312.exact80012RawTerms
def rightRaw : List Term := Proof.Events357.exact91477RawTerms
def group : MergeGroup := .operator 80012 91477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 80012) (leftOrdinal := 0)
    (rightResult := 91477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21904⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21904⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91483

namespace LeftMerge91604
def owner : Owner := ⟨.program ⟨214⟩, ⟨16423⟩⟩
def mergeEvent : Nat := 91604
def frameStart : Nat := 91538
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16381⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events357.exact91600RawTerms
def rightRaw : List Term := Proof.Events357.exact91598RawTerms
def group : MergeGroup := .operator 91600 91598
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91600) (leftOrdinal := 0)
    (rightResult := 91598) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16381⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91604

namespace LeftMerge91616
def owner : Owner := ⟨.program ⟨214⟩, ⟨28728⟩⟩
def mergeEvent : Nat := 91616
def frameStart : Nat := 91538
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩] } }
def leftRaw : List Term := Proof.Events357.exact91612RawTerms
def rightRaw : List Term := Proof.Events357.exact91589RawTerms
def group : MergeGroup := .operator 91612 91589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91612) (leftOrdinal := 0)
    (rightResult := 91589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28727⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91616

namespace LeftMerge91617
def owner : Owner := ⟨.program ⟨214⟩, ⟨28728⟩⟩
def mergeEvent : Nat := 91617
def frameStart : Nat := 91538
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16381⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩] } }
def leftRaw : List Term := Proof.Events357.exact91612RawTerms
def rightRaw : List Term := Proof.Events357.exact91589RawTerms
def group : MergeGroup := .operator 91612 91589
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91612) (leftOrdinal := 1)
    (rightResult := 91589) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16381⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28727⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91617

namespace LeftMerge91619
def owner : Owner := ⟨.program ⟨214⟩, ⟨28728⟩⟩
def mergeEvent : Nat := 91619
def frameStart : Nat := 91538
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16381⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24413⟩⟩] } }
def rhsRaw : List Term := Proof.Events357.exact91586RawTerms
def group : MergeGroup := .relation 91618
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91618) (rhsResult := 91586)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28727⟩⟩) ⟨24413⟩ 91586) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24413⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16381⟩⟩], [⟨.program ⟨214⟩, ⟨24413⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91619

namespace LeftMerge91627
def owner : Owner := ⟨.program ⟨214⟩, ⟨18841⟩⟩
def mergeEvent : Nat := 91627
def frameStart : Nat := 91538
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18832⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events357.exact91600RawTerms
def rightRaw : List Term := Proof.Events357.exact91623RawTerms
def group : MergeGroup := .operator 91600 91623
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 91600) (leftOrdinal := 0)
    (rightResult := 91623) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18832⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18832⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91627

namespace LeftMerge91644
def owner : Owner := ⟨.program ⟨214⟩, ⟨21907⟩⟩
def mergeEvent : Nat := 91644
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩] } }
def rhsRaw : List Term := Proof.Events357.exact91641RawTerms
def group : MergeGroup := .relation 91643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91643) (rhsResult := 91641)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21904⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 91642 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21904⟩⟩]⟩) (none) 91641) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6730⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6730⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge91644

namespace LeftMerge91645
def owner : Owner := ⟨.program ⟨214⟩, ⟨21907⟩⟩
def mergeEvent : Nat := 91645
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5507⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩] } }
def rhsRaw : List Term := Proof.Events357.exact91641RawTerms
def group : MergeGroup := .relation 91643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 91643) (rhsResult := 91641)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21904⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 91642 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21904⟩⟩]⟩) (none) 91641) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5507⟩⟩], [⟨.program ⟨214⟩, ⟨6701⟩⟩, ⟨.program ⟨214⟩, ⟨28727⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge91645

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
