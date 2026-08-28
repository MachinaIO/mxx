import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge57483
def owner : Owner := ⟨.program ⟨214⟩, ⟨12179⟩⟩
def mergeEvent : Nat := 57483
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events224.exact57477RawTerms
def rightRaw : List Term := Proof.Events052.exact13516RawTerms
def group : MergeGroup := .operator 57477 13516
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57477) (leftOrdinal := 1)
    (rightResult := 13516) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57483

namespace LeftMerge57485
def owner : Owner := ⟨.program ⟨214⟩, ⟨12179⟩⟩
def mergeEvent : Nat := 57485
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }
def rhsRaw : List Term := Proof.Events052.exact13486RawTerms
def group : MergeGroup := .relation 57484
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57484) (rhsResult := 13486)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7840⟩⟩) ⟨6775⟩ 13486) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57485

namespace LeftMerge57486
def owner : Owner := ⟨.program ⟨214⟩, ⟨12179⟩⟩
def mergeEvent : Nat := 57486
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events224.exact57477RawTerms
def rightRaw : List Term := Proof.Events052.exact13516RawTerms
def group : MergeGroup := .operator 57477 13516
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57477) (leftOrdinal := 0)
    (rightResult := 13516) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57486

namespace LeftMerge57491
def owner : Owner := ⟨.program ⟨214⟩, ⟨12180⟩⟩
def mergeEvent : Nat := 57491
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }
def leftRaw : List Term := Proof.Events224.exact57487RawTerms
def rightRaw : List Term := Proof.Events224.exact57457RawTerms
def group : MergeGroup := .operator 57487 57457
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57487) (leftOrdinal := 1)
    (rightResult := 57457) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6775⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6775⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57491

namespace LeftMerge57499
def owner : Owner := ⟨.program ⟨214⟩, ⟨25302⟩⟩
def mergeEvent : Nat := 57499
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩] } }
def leftRaw : List Term := Proof.Events224.exact57493RawTerms
def rightRaw : List Term := Proof.Events224.exact57429RawTerms
def group : MergeGroup := .operator 57493 57429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57493) (leftOrdinal := 1)
    (rightResult := 57429) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25301⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57499

namespace LeftMerge57501
def owner : Owner := ⟨.program ⟨214⟩, ⟨25302⟩⟩
def mergeEvent : Nat := 57501
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23166⟩⟩] } }
def rhsRaw : List Term := Proof.Events224.exact57426RawTerms
def group : MergeGroup := .relation 57500
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57500) (rhsResult := 57426)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25301⟩⟩) ⟨23166⟩ 57426) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23166⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩, ⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨23166⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57501

namespace LeftMerge57502
def owner : Owner := ⟨.program ⟨214⟩, ⟨25302⟩⟩
def mergeEvent : Nat := 57502
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩] } }
def leftRaw : List Term := Proof.Events224.exact57493RawTerms
def rightRaw : List Term := Proof.Events224.exact57429RawTerms
def group : MergeGroup := .operator 57493 57429
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57493) (leftOrdinal := 0)
    (rightResult := 57429) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25301⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57502

namespace LeftMerge57516
def owner : Owner := ⟨.program ⟨214⟩, ⟨19247⟩⟩
def mergeEvent : Nat := 57516
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19244⟩⟩] } }
def leftRaw : List Term := Proof.Events198.exact50762RawTerms
def rightRaw : List Term := Proof.Events224.exact57510RawTerms
def group : MergeGroup := .operator 50762 57510
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 50762) (leftOrdinal := 0)
    (rightResult := 57510) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19244⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57516

namespace LeftMerge57595
def owner : Owner := ⟨.program ⟨214⟩, ⟨12173⟩⟩
def mergeEvent : Nat := 57595
def frameStart : Nat := 57565
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events224.exact57591RawTerms
def rightRaw : List Term := Proof.Events224.exact57588RawTerms
def group : MergeGroup := .operator 57591 57588
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57591) (leftOrdinal := 0)
    (rightResult := 57588) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11137⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57595

namespace LeftMerge57625
def owner : Owner := ⟨.program ⟨214⟩, ⟨12276⟩⟩
def mergeEvent : Nat := 57625
def frameStart : Nat := 57565
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events225.exact57621RawTerms
def rightRaw : List Term := Proof.Events225.exact57619RawTerms
def group : MergeGroup := .operator 57621 57619
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57621) (leftOrdinal := 0)
    (rightResult := 57619) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57625

namespace LeftMerge57648
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def mergeEvent : Nat := 57648
def frameStart : Nat := 57565
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }
def leftRaw : List Term := Proof.Events225.exact57644RawTerms
def rightRaw : List Term := Proof.Events225.exact57641RawTerms
def group : MergeGroup := .operator 57644 57641
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57644) (leftOrdinal := 0)
    (rightResult := 57641) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7840⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57648

namespace LeftMerge57657
def owner : Owner := ⟨.program ⟨214⟩, ⟨25304⟩⟩
def mergeEvent : Nat := 57657
def frameStart : Nat := 57565
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩] } }
def leftRaw : List Term := Proof.Events225.exact57653RawTerms
def rightRaw : List Term := Proof.Events225.exact57610RawTerms
def group : MergeGroup := .operator 57653 57610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57653) (leftOrdinal := 0)
    (rightResult := 57610) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25301⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6792⟩⟩, ⟨.program ⟨214⟩, ⟨7840⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57657

namespace LeftMerge57658
def owner : Owner := ⟨.program ⟨214⟩, ⟨25304⟩⟩
def mergeEvent : Nat := 57658
def frameStart : Nat := 57565
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩] } }
def leftRaw : List Term := Proof.Events225.exact57653RawTerms
def rightRaw : List Term := Proof.Events225.exact57610RawTerms
def group : MergeGroup := .operator 57653 57610
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57653) (leftOrdinal := 1)
    (rightResult := 57610) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨25301⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57658

namespace LeftMerge57660
def owner : Owner := ⟨.program ⟨214⟩, ⟨25304⟩⟩
def mergeEvent : Nat := 57660
def frameStart : Nat := 57565
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨23166⟩⟩] } }
def rhsRaw : List Term := Proof.Events225.exact57607RawTerms
def group : MergeGroup := .relation 57659
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57659) (rhsResult := 57607)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨25301⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨25301⟩⟩) ⟨23166⟩ 57607) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨23166⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨11137⟩⟩, ⟨.program ⟨214⟩, ⟨12172⟩⟩], [⟨.program ⟨214⟩, ⟨23166⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge57660

namespace LeftMerge57668
def owner : Owner := ⟨.program ⟨214⟩, ⟨15428⟩⟩
def mergeEvent : Nat := 57668
def frameStart : Nat := 57565
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15426⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events225.exact57621RawTerms
def rightRaw : List Term := Proof.Events225.exact57664RawTerms
def group : MergeGroup := .operator 57621 57664
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 57621) (leftOrdinal := 0)
    (rightResult := 57664) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15426⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15426⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57668

namespace LeftMerge57685
def owner : Owner := ⟨.program ⟨214⟩, ⟨19247⟩⟩
def mergeEvent : Nat := 57685
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5513⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }
def rhsRaw : List Term := Proof.Events225.exact57682RawTerms
def group : MergeGroup := .relation 57684
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 57684) (rhsResult := 57682)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 57683 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19244⟩⟩]⟩) (none) 57682) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6693⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5513⟩⟩], [⟨.program ⟨214⟩, ⟨6693⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge57685

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
