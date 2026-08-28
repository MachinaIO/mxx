import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge46542
def owner : Owner := ⟨.program ⟨214⟩, ⟨30157⟩⟩
def mergeEvent : Nat := 46542
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24797⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46537RawTerms
def rightRaw : List Term := Proof.Events181.exact46359RawTerms
def group : MergeGroup := .operator 46537 46359
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46537) (leftOrdinal := 2)
    (rightResult := 46359) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24797⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24797⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨17019⟩⟩], [⟨.program ⟨214⟩, ⟨24797⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46542

namespace LeftMerge46550
def owner : Owner := ⟨.program ⟨214⟩, ⟨30158⟩⟩
def mergeEvent : Nat := 46550
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46544RawTerms
def rightRaw : List Term := Proof.Events021.exact5519RawTerms
def group : MergeGroup := .operator 46544 5519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46544) (leftOrdinal := 0)
    (rightResult := 5519) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6742⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6657⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6742⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46550

namespace LeftMerge46551
def owner : Owner := ⟨.program ⟨214⟩, ⟨30158⟩⟩
def mergeEvent : Nat := 46551
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46544RawTerms
def rightRaw : List Term := Proof.Events021.exact5519RawTerms
def group : MergeGroup := .operator 46544 5519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46544) (leftOrdinal := 1)
    (rightResult := 5519) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6657⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46551

namespace LeftMerge46553
def owner : Owner := ⟨.program ⟨214⟩, ⟨30158⟩⟩
def mergeEvent : Nat := 46553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events021.exact5512RawTerms
def group : MergeGroup := .relation 46552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46552) (rhsResult := 5512)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨6657⟩⟩) ⟨6600⟩ 5512) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨6435⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18132⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46553

namespace LeftMerge46567
def owner : Owner := ⟨.program ⟨214⟩, ⟨29840⟩⟩
def mergeEvent : Nat := 46567
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36805RawTerms
def rightRaw : List Term := Proof.Events181.exact46561RawTerms
def group : MergeGroup := .operator 36805 46561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36805) (leftOrdinal := 0)
    (rightResult := 46561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29838⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46567

namespace LeftMerge46568
def owner : Owner := ⟨.program ⟨214⟩, ⟨29840⟩⟩
def mergeEvent : Nat := 46568
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩] } }
def leftRaw : List Term := Proof.Events143.exact36805RawTerms
def rightRaw : List Term := Proof.Events181.exact46561RawTerms
def group : MergeGroup := .operator 36805 46561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36805) (leftOrdinal := 1)
    (rightResult := 46561) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29838⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46568

namespace LeftMerge46570
def owner : Owner := ⟨.program ⟨214⟩, ⟨29840⟩⟩
def mergeEvent : Nat := 46570
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24734⟩⟩] } }
def rhsRaw : List Term := Proof.Events181.exact46558RawTerms
def group : MergeGroup := .relation 46569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46569) (rhsResult := 46558)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29838⟩⟩) ⟨24734⟩ 46558) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24734⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨24734⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46570

namespace LeftMerge46584
def owner : Owner := ⟨.program ⟨214⟩, ⟨22635⟩⟩
def mergeEvent : Nat := 46584
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22632⟩⟩] } }
def leftRaw : List Term := Proof.Events141.exact36137RawTerms
def rightRaw : List Term := Proof.Events181.exact46578RawTerms
def group : MergeGroup := .operator 36137 46578
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 36137) (leftOrdinal := 0)
    (rightResult := 46578) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22632⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22632⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46584

namespace LeftMerge46705
def owner : Owner := ⟨.program ⟨214⟩, ⟨16977⟩⟩
def mergeEvent : Nat := 46705
def frameStart : Nat := 46639
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46701RawTerms
def rightRaw : List Term := Proof.Events182.exact46699RawTerms
def group : MergeGroup := .operator 46701 46699
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46701) (leftOrdinal := 0)
    (rightResult := 46699) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46705

namespace LeftMerge46717
def owner : Owner := ⟨.program ⟨214⟩, ⟨29839⟩⟩
def mergeEvent : Nat := 46717
def frameStart : Nat := 46639
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46713RawTerms
def rightRaw : List Term := Proof.Events182.exact46690RawTerms
def group : MergeGroup := .operator 46713 46690
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46713) (leftOrdinal := 0)
    (rightResult := 46690) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29838⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46717

namespace LeftMerge46718
def owner : Owner := ⟨.program ⟨214⟩, ⟨29839⟩⟩
def mergeEvent : Nat := 46718
def frameStart : Nat := 46639
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46713RawTerms
def rightRaw : List Term := Proof.Events182.exact46690RawTerms
def group : MergeGroup := .operator 46713 46690
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46713) (leftOrdinal := 1)
    (rightResult := 46690) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨29838⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46718

namespace LeftMerge46720
def owner : Owner := ⟨.program ⟨214⟩, ⟨29839⟩⟩
def mergeEvent : Nat := 46720
def frameStart : Nat := 46639
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24734⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46687RawTerms
def group : MergeGroup := .relation 46719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46719) (rhsResult := 46687)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨29838⟩⟩) ⟨24734⟩ 46687) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24734⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨24734⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46720

namespace LeftMerge46728
def owner : Owner := ⟨.program ⟨214⟩, ⟨16937⟩⟩
def mergeEvent : Nat := 46728
def frameStart : Nat := 46639
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16935⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events182.exact46701RawTerms
def rightRaw : List Term := Proof.Events182.exact46724RawTerms
def group : MergeGroup := .operator 46701 46724
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46701) (leftOrdinal := 0)
    (rightResult := 46724) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16935⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16935⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46728

namespace LeftMerge46745
def owner : Owner := ⟨.program ⟨214⟩, ⟨22635⟩⟩
def mergeEvent : Nat := 46745
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46742RawTerms
def group : MergeGroup := .relation 46744
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46744) (rhsResult := 46742)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46743 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22632⟩⟩]⟩) (none) 46742) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6740⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6740⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46745

namespace LeftMerge46746
def owner : Owner := ⟨.program ⟨214⟩, ⟨22635⟩⟩
def mergeEvent : Nat := 46746
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46742RawTerms
def group : MergeGroup := .relation 46744
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46744) (rhsResult := 46742)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46743 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22632⟩⟩]⟩) (none) 46742) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6706⟩⟩, ⟨.program ⟨214⟩, ⟨29838⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46746

namespace LeftMerge46747
def owner : Owner := ⟨.program ⟨214⟩, ⟨22635⟩⟩
def mergeEvent : Nat := 46747
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24734⟩⟩] } }
def rhsRaw : List Term := Proof.Events182.exact46742RawTerms
def group : MergeGroup := .relation 46744
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46744) (rhsResult := 46742)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22632⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 46743 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨22632⟩⟩]⟩) (none) 46742) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16879⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24734⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5515⟩⟩, ⟨.program ⟨214⟩, ⟨16879⟩⟩], [⟨.program ⟨214⟩, ⟨24734⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge46747

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
