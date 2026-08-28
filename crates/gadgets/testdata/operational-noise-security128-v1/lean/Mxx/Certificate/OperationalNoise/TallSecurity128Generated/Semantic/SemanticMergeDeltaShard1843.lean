import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge297780
def owner : Owner := ⟨.program ⟨257⟩, ⟨28541⟩⟩
def mergeEvent : Nat := 297780
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events1163.exact297776RawTerms
def rightRaw : List Term := Proof.Events1163.exact297746RawTerms
def group : MergeGroup := .operator 297776 297746
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297776) (leftOrdinal := 1)
    (rightResult := 297746) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297780

namespace LeftMerge297788
def owner : Owner := ⟨.program ⟨257⟩, ⟨30490⟩⟩
def mergeEvent : Nat := 297788
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩] } }
def leftRaw : List Term := Proof.Events1163.exact297782RawTerms
def rightRaw : List Term := Proof.Events1162.exact297718RawTerms
def group : MergeGroup := .operator 297782 297718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297782) (leftOrdinal := 1)
    (rightResult := 297718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30489⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297788

namespace LeftMerge297790
def owner : Owner := ⟨.program ⟨257⟩, ⟨30490⟩⟩
def mergeEvent : Nat := 297790
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30029⟩⟩] } }
def rhsRaw : List Term := Proof.Events1162.exact297715RawTerms
def group : MergeGroup := .relation 297789
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297789) (rhsResult := 297715)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30489⟩⟩) ⟨30029⟩ 297715) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30029⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], [⟨.program ⟨257⟩, ⟨30029⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297790

namespace LeftMerge297791
def owner : Owner := ⟨.program ⟨257⟩, ⟨30490⟩⟩
def mergeEvent : Nat := 297791
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩] } }
def leftRaw : List Term := Proof.Events1163.exact297782RawTerms
def rightRaw : List Term := Proof.Events1162.exact297718RawTerms
def group : MergeGroup := .operator 297782 297718
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297782) (leftOrdinal := 0)
    (rightResult := 297718) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30489⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297791

namespace LeftMerge297805
def owner : Owner := ⟨.program ⟨257⟩, ⟨29432⟩⟩
def mergeEvent : Nat := 297805
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩] } }
def leftRaw : List Term := Proof.Events1153.exact295195RawTerms
def rightRaw : List Term := Proof.Events1163.exact297799RawTerms
def group : MergeGroup := .operator 295195 297799
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 295195) (leftOrdinal := 0)
    (rightResult := 297799) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨29429⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297805

namespace LeftMerge297860
def owner : Owner := ⟨.program ⟨257⟩, ⟨28535⟩⟩
def mergeEvent : Nat := 297860
def frameStart : Nat := 297842
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events1163.exact297856RawTerms
def rightRaw : List Term := Proof.Events1163.exact297853RawTerms
def group : MergeGroup := .operator 297856 297853
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297856) (leftOrdinal := 0)
    (rightResult := 297853) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13131⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297860

namespace LeftMerge297890
def owner : Owner := ⟨.program ⟨257⟩, ⟨30328⟩⟩
def mergeEvent : Nat := 297890
def frameStart : Nat := 297842
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1163.exact297886RawTerms
def rightRaw : List Term := Proof.Events1163.exact297884RawTerms
def group : MergeGroup := .operator 297886 297884
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297886) (leftOrdinal := 0)
    (rightResult := 297884) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297890

namespace LeftMerge297913
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def mergeEvent : Nat := 297913
def frameStart : Nat := 297842
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events1163.exact297909RawTerms
def rightRaw : List Term := Proof.Events1163.exact297906RawTerms
def group : MergeGroup := .operator 297909 297906
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297909) (leftOrdinal := 0)
    (rightResult := 297906) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297913

namespace LeftMerge297922
def owner : Owner := ⟨.program ⟨257⟩, ⟨30492⟩⟩
def mergeEvent : Nat := 297922
def frameStart : Nat := 297842
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩] } }
def leftRaw : List Term := Proof.Events1163.exact297918RawTerms
def rightRaw : List Term := Proof.Events1163.exact297875RawTerms
def group : MergeGroup := .operator 297918 297875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297918) (leftOrdinal := 0)
    (rightResult := 297875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30489⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297922

namespace LeftMerge297923
def owner : Owner := ⟨.program ⟨257⟩, ⟨30492⟩⟩
def mergeEvent : Nat := 297923
def frameStart : Nat := 297842
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩] } }
def leftRaw : List Term := Proof.Events1163.exact297918RawTerms
def rightRaw : List Term := Proof.Events1163.exact297875RawTerms
def group : MergeGroup := .operator 297918 297875
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297918) (leftOrdinal := 1)
    (rightResult := 297875) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30489⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297923

namespace LeftMerge297925
def owner : Owner := ⟨.program ⟨257⟩, ⟨30492⟩⟩
def mergeEvent : Nat := 297925
def frameStart : Nat := 297842
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30029⟩⟩] } }
def rhsRaw : List Term := Proof.Events1163.exact297872RawTerms
def group : MergeGroup := .relation 297924
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297924) (rhsResult := 297872)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30489⟩⟩) ⟨30029⟩ 297872) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30029⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], [⟨.program ⟨257⟩, ⟨30029⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297925

namespace LeftMerge297933
def owner : Owner := ⟨.program ⟨257⟩, ⟨29010⟩⟩
def mergeEvent : Nat := 297933
def frameStart : Nat := 297842
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1163.exact297886RawTerms
def rightRaw : List Term := Proof.Events1163.exact297929RawTerms
def group : MergeGroup := .operator 297886 297929
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 297886) (leftOrdinal := 0)
    (rightResult := 297929) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29008⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297933

namespace LeftMerge297950
def owner : Owner := ⟨.program ⟨257⟩, ⟨29432⟩⟩
def mergeEvent : Nat := 297950
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events1163.exact297947RawTerms
def group : MergeGroup := .relation 297949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297949) (rhsResult := 297947)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297948 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩]⟩) (none) 297947) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297950

namespace LeftMerge297951
def owner : Owner := ⟨.program ⟨257⟩, ⟨29432⟩⟩
def mergeEvent : Nat := 297951
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩] } }
def rhsRaw : List Term := Proof.Events1163.exact297947RawTerms
def group : MergeGroup := .relation 297949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297949) (rhsResult := 297947)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297948 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩]⟩) (none) 297947) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30489⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297951

namespace LeftMerge297952
def owner : Owner := ⟨.program ⟨257⟩, ⟨29432⟩⟩
def mergeEvent : Nat := 297952
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30029⟩⟩] } }
def rhsRaw : List Term := Proof.Events1163.exact297947RawTerms
def group : MergeGroup := .relation 297949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297949) (rhsResult := 297947)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297948 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩]⟩) (none) 297947) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30029⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨13131⟩⟩, ⟨.program ⟨257⟩, ⟨28534⟩⟩], [⟨.program ⟨257⟩, ⟨30029⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge297952

namespace LeftMerge297953
def owner : Owner := ⟨.program ⟨257⟩, ⟨29432⟩⟩
def mergeEvent : Nat := 297953
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1163.exact297947RawTerms
def group : MergeGroup := .relation 297949
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 297949) (rhsResult := 297947)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 297948 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29429⟩⟩]⟩) (none) 297947) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29008⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2377⟩⟩, ⟨.program ⟨257⟩, ⟨29008⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge297953

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
